import pandas as pd
import json
import argparse
import os
from datasets import load_dataset, Features, Value, DatasetDict, Dataset
from transformers import set_seed

set_seed(42)

parser = argparse.ArgumentParser(prog="Dataset Preparation")
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument("--dataset_path", type=str, default="./data")
parser.add_argument("--output_path", type=str, default="./processed_data")
parser.add_argument("--language", type=str, default="en")
args = parser.parse_args()


### CLEF_3A ###
def prepare_clef3a(dataset_path):
    data_files = {
        "train": f"{dataset_path}/{args.dataset_name}/train.tsv",
        "validation": f"{dataset_path}/{args.dataset_name}/dev.tsv",
        "test": f"{dataset_path}/{args.dataset_name}/test.tsv",
    }
    return load_dataset("csv", data_files=data_files, delimiter="\t")


### CLEF_1C ###
def prepare_clef1c(dataset_path, language):
    features = Features(
        {
            "tweet_text": Value("string"),
            "class_label": Value("int64"),
        }
    )

    data_files = {
        "train": f"{dataset_path}/{args.dataset_name}/CT22_{language.lower()}_1C_harmful_train.tsv",
        "test": f"{dataset_path}/{args.dataset_name}/CT22_{language.lower()}_1C_harmful_test_gold.tsv",
    }

    dataset = load_dataset(
        "csv", data_files=data_files, delimiter="\t", features=features
    )
    dataset = dataset.rename_column("class_label", "label")
    dataset = dataset.rename_column("tweet_text", "text")
    return dataset


### ALL-SIDES ###
def prepare_allsides(dataset_path):
    labels = ["left", "center", "right"]
    dataset = load_dataset(
        "csv",
        data_files=f"{dataset_path}/{args.dataset_name}/allsides_balanced_news_headlines-texts.csv",
    )
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    def concatenate_header_text(example):
        example["text"] = f"{example['heading']}\n{example['text']}"
        example["label"] = labels.index(example["bias_rating"])
        return example

    return dataset.map(concatenate_header_text)


### FAKE-BR ###
def prepare_fakebr_corpus(dataset_path):
    dataset = load_dataset(
        "csv",
        data_files=f"{dataset_path}/{args.dataset_name}/Fake.br-Corpus.tsv",
        delimiter="\t",
    )
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

    def format_func(element):
        element["label"] = ["fake", "true"].index(element["label"])
        return element

    return dataset.map(format_func)


### FAKE NEWS CORPUS SPANISH ###
def prepare_fake_news_corpus_spanish(dataset_path):
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(
                pd.read_excel(f"{dataset_path}/{args.dataset_name}/train.xlsx")
            ),
            "dev": Dataset.from_pandas(
                pd.read_excel(f"{dataset_path}/{args.dataset_name}/development.xlsx")
            ),
            "test": Dataset.from_pandas(
                pd.read_excel(f"{dataset_path}/{args.dataset_name}/test.xlsx")
            ),
        }
    )

    dataset["test"] = dataset["test"].rename_columns(
        {
            "ID": "Id",
            "CATEGORY": "Category",
            "TOPICS": "Topic",
            "SOURCE": "Source",
            "HEADLINE": "Headline",
            "TEXT": "Text",
            "LINK": "Link",
        }
    )

    def format_func2(element):
        labels = ["Fake", "True"]
        bool_labels = [False, True]
        element["text"] = f"{element['Headline']}\n{element['Text']}"
        if isinstance(element["Category"], str):
            element["label"] = labels.index(element["Category"])
        elif isinstance(element["Category"], bool):
            element["label"] = bool_labels.index(element["Category"])
        return element

    return dataset.map(format_func2)


### FAKE NEWS NET ###
def prepare_fake_news_net(dataset_path):
    fake_split = [
        f"{dataset_path}/{args.dataset_name}/politifact_fake.csv",
        f"{dataset_path}/{args.dataset_name}/gossipcop_fake.csv",
    ]
    real_split = [
        f"{dataset_path}/{args.dataset_name}/politifact_real.csv",
        f"{dataset_path}/{args.dataset_name}/gossipcop_real.csv",
    ]

    fakes = load_dataset("csv", data_files=fake_split, split="train").to_list()
    real = load_dataset("csv", data_files=real_split, split="train").to_list()

    for item in fakes:
        item["label"] = 0
    for item in real:
        item["label"] = 1

    dataset = Dataset.from_list(fakes + real)
    dataset = dataset.rename_column("title", "text")
    dataset = dataset.remove_columns(["id", "news_url", "tweet_ids"])
    return dataset.train_test_split(test_size=0.2, seed=42)


### HYPERPARTISAN NEWS HEADLINES ###
def prepare_hyperpartisan_news_headlines(dataset_path):
    data_files = {
        "train": f"{dataset_path}/{args.dataset_name}/HP_news_title_train.csv",
        "test": f"{dataset_path}/{args.dataset_name}/HP_news_title_test.csv",
    }
    dataset = load_dataset("csv", data_files=data_files)
    return dataset.rename_column("title", "text")


### SEMEVAL2019 ###
def prepare_semeval(dataset_path):
    data_files = {
        "train": f"{dataset_path}/{args.dataset_name}/train.tsv",
        "test": f"{dataset_path}/{args.dataset_name}/test.tsv",
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    return dataset.rename_column("sentence", "text")


### MAIN FUNCTION ###
def main():
    dataset = None

    if args.dataset_name.lower() == "clef_3a":
        dataset = prepare_clef3a(args.dataset_path)
    elif args.dataset_name.lower() == "clef_1c":
        dataset = prepare_clef1c(args.dataset_path, args.language)
    elif args.dataset_name.lower() == "all_sides":
        dataset = prepare_allsides(args.dataset_path)
    elif args.dataset_name.lower() == "fake_br_corpus":
        dataset = prepare_fakebr_corpus(args.dataset_path)
    elif args.dataset_name.lower() == "fake_news_corpus_spanish":
        dataset = prepare_fake_news_corpus_spanish(args.dataset_path)
    elif args.dataset_name.lower() == "fake_news_net":
        dataset = prepare_fake_news_net(args.dataset_path)
    elif args.dataset_name.lower() == "hyperpartisan_news_headlines":
        dataset = prepare_hyperpartisan_news_headlines(args.dataset_path)
    elif args.dataset_name.lower() == "semeval_2019":
        dataset = prepare_semeval(args.dataset_path)
    else:
        print(f"Unknown dataset name: {args.dataset_name}")
        return

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Save each split to JSON
    for split in dataset:
        print(f"Saving {split} split...")
        dataset[split].to_json(f"{args.output_path}/{split}.json")


if __name__ == "__main__":
    main()
