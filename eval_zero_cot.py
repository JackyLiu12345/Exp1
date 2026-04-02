from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import load_dataset
import argparse
import torch
import time
import random
import json
import re
import wandb
from utils import compute_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(42)

# --- Params parsing
parser = argparse.ArgumentParser(prog="Zero-Shot Evaluation Script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--dataset_name", type=str)
parser.add_argument(
    "--configuration",
    type=str,
    default="zero_shot_generic",
    choices=["zero_shot_generic", "zero_shot_specific", "codebook", "cot"],
    help="Evaluation configuration: zero-shot variants",
)
parser.add_argument(
    "--language",
    type=str,
    default="en",
    help="Language for prompts ('en', 'bg', 'ar', 'es', 'pt')",
)
parser.add_argument(
    "--task_labels", type=str, default="", choices=["hp", "pl", "ht", "fn"]
)
parser.add_argument(
    "--label_type",
    type=str,
    default="string",
    choices=["string", "int"],
    help="Format of the labels in the model's output: 'string' for text labels or 'int' for integer labels",
)
args = parser.parse_args()

print("Parsed Arguments:", args)

main_run = wandb.init(
    project=args.dataset_name,
    entity="",
    name=f"[{args.configuration}] [{args.label_type}] {args.model_name.split('/')[1]}_zero_shot",
    reinit=True,
    config={
        "quantization": args.use_quantization,
        "language": args.language,
        "task_labels": args.task_labels,
        "label_type": args.label_type,
        "configuration": args.configuration,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "verbose": args.verbose,
    },
)

# ---- Model/Tokenizer loading
if args.use_quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    quantization_config = None

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)

model.config.use_cache = True

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.generation_config.pad_token_id = tokenizer.pad_token_id
print(model.generation_config)

# ---- Dataset loading/Processing
dataset = load_dataset(
    "json",
    data_files={
        "train": "./processed_data/train.json",
        "test": "./processed_data/test.json",
    },
)
print(dataset)

num_labels = len(dataset["train"].unique("label"))
print(" > Label num: ", num_labels)

if args.task_labels == "hp":
    labels = ["neutral", "hyperpartisan"]
elif args.task_labels == "fn":
    if args.language == "en":
        labels = ["true", "fake"]
    elif args.language == "es":
        labels = ["verdadera", "falsa"]
    elif args.language == "pt":
        labels = ["verdadera", "falsa"]
elif args.task_labels == "ht":
    if args.language == "en":
        labels = ["neutral", "harmful"]
    elif args.language == "bg":
        labels = ["неутрално", "вредно"]
    elif args.language == "ar":
        labels = ["محايد", "ضار"]
elif args.task_labels == "pl":
    labels = ["left", "center", "right"]


# Load prompts for zero-shot evaluation
if args.label_type == "string":
    prompt_path = "./prompts_ICWSM_str.json"
else:
    prompt_path = "./prompts_ICWSM_int.json"

with open(prompt_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)

prompt = prompts[args.task_labels][args.language][args.configuration]
print("-" * 10, "Prompt", "-" * 10)
print(f"Using prompt: {prompt}")
print("-" * 25)


def parse_label(model_output):
    # match = re.search(r"(?:\s*==>|\s*:)\s*(?:\*\*)?([^\s*]+)(?:\*\*)?", model_output)
    # match = re.search(r"==>\s*([^\s]+)", model_output)
    match = re.search(r"==>\s*(\w+)", model_output)

    if not match:
        return None

    content = match.group(1).strip().lower()

    if args.label_type == "int":
        if "0" in content:
            return 0
        elif "1" in content:
            return 1
        elif args.task_labels == "pl" and "2" in content:
            return 2
        else:
            return None

    if args.task_labels == "hp":
        if content == "neutral":
            return 0
        elif content == "hyperpartisan":
            return 1

    elif args.task_labels == "fn":
        if content in ["true", "verdadera", "verdadeira"]:
            return 0
        elif content in ["fake", "false", "falsa", "falsa"]:
            return 1

    elif args.task_labels == "ht":
        if content in ["neutral", "محايد", "неутрално", "неутрален"]:
            return 0
        elif content in ["harmful", "ضار", "вредно", "вреден"]:
            return 1

    elif args.task_labels == "pl":
        if content in ["center", "neutral"]:
            return 1
        elif content == "left":
            return 0
        elif content == "right":
            return 2

    return None


def generate(model, tokenizer, prompt, element, do_sample=False, temperature=0.0):
    formatted_prompt = prompt.format(element)

    messages = [
        {"role": "user", "content": formatted_prompt},
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=do_sample,
        temperature=temperature,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output


# ---- Inference
results = {}
model_outputs = {}

irregular_outputs = 0
regularized_outputs = 0
skipped_items = 0

preds = []
refs = []

start_time = time.time()
for idx, element in enumerate(dataset["test"]):
    pred = generate(model, tokenizer, prompt, element["text"])

    args.verbose and print("=========== Model output ===========")
    args.verbose and print(pred)
    args.verbose and print("====================================")

    parsed_label = parse_label(pred)
    print(f" > Pred #{idx}: ", parsed_label)
    print(f" > Ref #{idx}: ", element["label"])

    if parsed_label is None:
        irregular_outputs += 1
        max_retries = 5
        retry_count = 0

        print(f" > Irregular output at element #{idx}:  ", pred)
        print("*" * 5, "Trying to resolve irregularity", "*" * 5)

        while retry_count < max_retries:
            pred = generate(
                model,
                tokenizer,
                prompt,
                element["text"],
                do_sample=True,
                temperature=0.7,
            )
            args.verbose and print(f" >> Attempted Pred: {pred}")

            if parse_label(pred) is not None:
                print(" >> Regularized output: ", pred)
                parsed_label = parse_label(pred)
                print(f" > Pred #{idx}: ", parsed_label)

                regularized_outputs += 1
                break

            retry_count += 1

            if retry_count == args.max_retries:
                print(
                    " >> Failed to get valid prediction after max retries.\n > Forcefully considered false prediction."
                )
                skipped_items += 1

                fallback_label_int = (element["label"] + 1) % num_labels
                preds.append(fallback_label_int)
                refs.append(element["label"])
                continue

    preds.append(parsed_label)
    refs.append(element["label"])

results = compute_metrics(preds, refs)
results["irregular_outputs"] = irregular_outputs
results["regularized_outputs"] = regularized_outputs
results["skipped_items"] = skipped_items

model_outputs = {
    "ground_truth": refs,
    "model_predictions": preds,
}

print(json.dumps(results, indent=4))

for metric in results:
    main_run.log({f"avg_{metric}": results[metric]})

print(f" > Inference execution time: {(time.time() - start_time):.2f}s")

# ---- Saving results/outputs to JSON files
with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)
main_run.save("results.json")

with open("model_outputs.json", "w") as json_file:
    json.dump(model_outputs, json_file, indent=4)
main_run.save("model_outputs.json")

main_run.finish()
