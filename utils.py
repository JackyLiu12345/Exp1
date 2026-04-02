"""Utility functions for dataset analysis and metric computation.

Provides helpers for computing token length statistics, evaluation metrics
(accuracy, precision, recall, F1), and aggregating results across runs.
"""

import numpy as np
import evaluate


def get_dataset_length_stats(tokenizer, dataset):
    result = {}

    for split in dataset:
        lengths = []
        for element in dataset[split]:
            tokens = tokenizer(element["text"], return_tensors="pt")
            lengths.append(tokens["input_ids"].shape[1])

        result[split] = {
            "max": int(np.max(lengths)),
            "min": int(np.min(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
        }

    return result


def compute_metrics(predictions, labels):
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["precision"]
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["recall"]
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_average_metrics(data):
    result = {}
    metrics = data[0].keys()

    for metric in metrics:
        scores = [element[metric] for element in data]
        avg = np.mean(scores)
        minimum = np.min(scores)
        maximum = np.max(scores)
        stddev = np.std(scores)

        result[metric] = {
            "score": avg,
            "stddev": stddev,
            "min": float(minimum),
            "max": float(maximum),
        }

    return result


def compute_few_shot_nested_avg(nested_evals):
    avg_evals = {}
    shots = list(nested_evals[list(nested_evals.keys())[0]].keys())

    for shot in shots:
        aggregate_evals = []

        for nested_result in nested_evals:
            aggregate_evals.append(nested_evals[nested_result][shot])

        evals = compute_average_metrics(aggregate_evals)
        avg_evals[shot] = evals

    return avg_evals
