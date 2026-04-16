"""Few-shot in-context learning evaluation script for LLMs.

Evaluates language models using DPP-selected or randomly sampled few-shot examples
for news classification tasks.

Usage:
    python eval_few_shot.py --model_name MODEL --dataset_name DATASET --configuration fs_dpp --task_labels LABELS
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from datasets import load_dataset
import argparse
import torch
import random
import json
import re
import wandb
from utils import compute_metrics, compute_average_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
set_seed(42)

# --- Params parsing
parser = argparse.ArgumentParser(prog="Few-Shot Evaluation Script")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
)
parser.add_argument("--use_quantization", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument("--max_retries", type=int, default=5)
parser.add_argument("--runs", type=int, default=5)
parser.add_argument(
    "--configuration",
    type=str,
    default="fs_dpp",
    choices=["fs_dpp", "fs_random"],
    help="Evaluation configuration: fs_dpp (DPP few-shot) or fs_random (random few-shot)",
)
parser.add_argument(
    "--language", type=str, default="en", help="Language for prompts ('bg', 'en', 'pt')"
)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=0)
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

# Validate language-task combinations
valid_languages = {
    "hp": ["en"],
    "pl": ["en"],
    "fn": ["en", "es", "pt"],
    "ht": ["en", "bg", "ar"],
}
if args.task_labels in valid_languages and args.language not in valid_languages[args.task_labels]:
    parser.error(
        f"Language '{args.language}' is not supported for task '{args.task_labels}'. "
        f"Supported: {valid_languages[args.task_labels]}"
    )

print(args)


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
    attn_implementation="sdpa",
    quantization_config=quantization_config,
)

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
# print(dataset["train"][0])

num_labels = len(dataset["train"].unique("label"))
print(" > Label num: ", num_labels)

# ---- Prompts loading / Labels parsing logic
if args.label_type == "string":
    prompt_path = "./prompts_ICWSM_str.json"
else:
    prompt_path = "./prompts_ICWSM_int.json"

with open(prompt_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)

prompt = prompts[args.task_labels][args.language]["few_shot"]
print("-" * 10, "Prompt", "-" * 10)
print(f"Using prompt: {prompt}")
print("-" * 25)

# Load DPP few-shot examples if using the DPP method
if args.configuration == "fs_dpp":
    if args.dataset_name == "clef_1c":
        with open(
            f"./data/{args.dataset_name}/{args.dataset_name}_{args.language}_5_runs.json"
        ) as json_file:
            few_shot_examples = json.load(json_file)
    else:
        with open(
            f"./data/{args.dataset_name}/{args.dataset_name}_5_runs.json"
        ) as json_file:
            few_shot_examples = json.load(json_file)

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


def parse_label(model_output):
    match = re.search(r"==>\s*(.+)", model_output)

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


def generate(
    model,
    tokenizer,
    prompt,
    few_shot_examples,
    element,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
):
    formatted_prompt = prompt.format(few_shot_examples, element)

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
        max_new_tokens=256,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output


def construct_few_shot_string_dpp(example_set, few_shot_n, num_labels):
    few_shot_string = ""

    examples_per_label = few_shot_n // num_labels

    for label in example_set:
        for n in range(examples_per_label):
            int_label = int(re.search(r"\d+$", label).group())
            example = example_set[label][n]
            example = re.sub(r",\s*(?:'\d+'|\d+)$", f"==> {labels[int_label]}", example)
            few_shot_string += example + "\n\n"

    return few_shot_string


def construct_few_shot_string_random(dataset, n_shot, seed):
    random.seed(seed)

    few_shot_examples = []
    dataset_labels = list(set(example["label"] for example in dataset["train"]))

    for shot in range(n_shot):
        # ---- Selecting a unique example from the dataset (alternating labels)
        shot_label = shot % len(dataset_labels)

        filtered_dataset = dataset["train"].filter(
            lambda example: example["label"] == shot_label
        )
        while True:
            random_index = random.randint(0, len(filtered_dataset) - 1)
            random_element = filtered_dataset[random_index]
            if random_element not in few_shot_examples:
                break

        few_shot_examples.append(random_element)

    few_shot_string = ""
    for item in few_shot_examples:
        few_shot_string += f"{item['text']} ==> {labels[item['label']]}\n\n"

    return few_shot_string


if args.configuration == "fs_dpp":
    start = args.start if args.start != 0 else num_labels
    end = args.end + 1 if args.end != 0 else 11
    step = num_labels

if args.configuration == "fs_random":
    start = args.start if args.start != 0 else 1
    end = args.end + 1 if args.end != 0 else 11
    step = 1

seeds = [42, 12345, 9876, 2024, 8675309]
for few_shot in range(start, end, step):
    wandb.init(
        project=args.dataset_name,
        entity="",
        name=f"[{args.configuration}][{few_shot}_shot] {args.model_name.split('/')[1]}",
        reinit=True,
        config={
            "quantization": args.use_quantization,
            "language": args.language,
            "task_labels": args.task_labels,
            "label_type": args.label_type,
            "configuration": args.configuration,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "num_runs": args.runs,
            "verbose": args.verbose,
        },
    )

    print(f" => Evaluating {few_shot}-shot")
    run_evals = []
    for run in range(args.runs):
        if args.configuration == "fs_dpp":
            examples = few_shot_examples[f"Setn_{run + 1}"]
            few_shot_string = construct_few_shot_string_dpp(
                examples,
                few_shot,
                num_labels,
            )

        if args.configuration == "fs_random":
            few_shot_string = construct_few_shot_string_random(
                dataset, few_shot, seeds[run]
            )

        args.verbose and print("======= Few-shot string =======")
        args.verbose and print(few_shot_string)
        args.verbose and print("===============================")

        preds = []
        refs = []
        unparseable_outputs = []

        irregular_outputs = 0
        regularized_outputs = 0
        skipped_items = 0

        for idx, element in enumerate(dataset["test"]):
            pred = generate(
                model,
                tokenizer,
                prompt,
                few_shot_string,
                element["text"],
            )

            args.verbose and print("=========== Model output ===========")
            args.verbose and print(pred)
            args.verbose and print("====================================")

            parsed_label = parse_label(pred)
            print(f" > Pred #{idx}: ", parsed_label)
            print(f" > Ref #{idx}: ", element["label"])

            if parsed_label is None:
                irregular_outputs += 1
                print(" > Unparsable output")

                print("*" * 5, "Trying to resolve irregularity", "*" * 5)
                retry_count = 0
                while retry_count < args.max_retries:
                    pred = generate(
                        model,
                        tokenizer,
                        prompt,
                        few_shot_string,
                        element["text"],
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    print(" >> Attempted Pred: ", pred)

                    if parse_label(pred) is not None:
                        regularized_outputs += 1
                        print(" >> Regularized output: ", pred)
                        break

                    retry_count += 1

                if retry_count == args.max_retries:
                    print(
                        " >> Failed to get valid prediction after max retries.\n > Forcefully considered false prediction."
                    )
                    skipped_items += 1
                    unparseable_outputs.append({"index": idx, "run": run, "output": pred, "ground_truth": element["label"]})

                    fallback_label_int = random.randint(0, num_labels - 1)
                    preds.append(fallback_label_int)
                    refs.append(element["label"])
                    continue

            preds.append(parse_label(pred))
            refs.append(element["label"])

        evals = compute_metrics(preds, refs)
        print(json.dumps(evals, indent=4))

        if unparseable_outputs:
            print(f" > {len(unparseable_outputs)} unparseable outputs in run {run}")

        run_evals.append(evals)

    avg_run_evals = compute_average_metrics(run_evals)
    print(json.dumps(avg_run_evals, indent=4))

    for metric in avg_run_evals:
        wandb.log({f"avg_{metric}": avg_run_evals[metric]["score"]})
    with open("avg_results.json", "w") as json_file:
        json.dump(avg_run_evals, json_file, indent=4)
    wandb.save("avg_results.json")

    wandb.log(
        {
            "irregular_outputs": irregular_outputs,
            "regularized_outputs": regularized_outputs,
            "skipped_items": skipped_items,
        }
    )

    wandb.finish()
