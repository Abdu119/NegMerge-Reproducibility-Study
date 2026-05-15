# NegMerge
# Copyright (c) 2025-present NAVER Cloud Corp.
# MIT license 
# Referenced from: https://github.com/gortizji/tangent_task_arithmetic/blob/main/src/eval_task_negation.py 


import torch
import os
from src.args import parse_arguments
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from utils import find_optimal_coef
import json

merge_datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]

args = parse_arguments()

args.save = f"{args.results_db}/{args.finetuning_mode}/{args.model}"

with open(os.path.join(f"{args.save}", "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

control_dataset = "ImageNet"
negation_accuracies = {}

for dataset in merge_datasets:

    pretrained_checkpoint = (
        f"{args.save}/linear_zeroshot.pt"
        if args.finetuning_mode == "linear"
        else f"{args.save}/zeroshot.pt"
    )

    checkpoint_paths = []
    for dir_name in os.listdir(args.save):
        dir_path = os.path.join(args.save, dir_name)
        if os.path.isdir(dir_path) and "checkpoints" in dir_name:
            if f"{dataset}Val" in os.listdir(dir_path):
                val_dir_path = os.path.join(dir_path, f"{dataset}Val")
                if "finetuned.pt" in os.listdir(val_dir_path):
                    checkpoint_paths.append(os.path.join(val_dir_path, "finetuned.pt"))

    for idx, checkpoint_path in enumerate(checkpoint_paths):
        task_vector = (
            LinearizedTaskVector(pretrained_checkpoint, checkpoint_path)
            if args.finetuning_mode == "linear"
            else NonLinearTaskVector(pretrained_checkpoint, checkpoint_path)
        )

        if idx == 0:
            merged_vector = {k: torch.zeros_like(v) for k, v in task_vector.vector.items()}
            mask = {k: torch.zeros_like(v) for k, v in task_vector.vector.items()}

        for key in task_vector.vector.keys():
            merged_vector[key] += task_vector.vector[key]
            mask[key] += torch.sign(task_vector.vector[key])

    for key in torch.load(checkpoint_path).state_dict().keys():
        consistency_mask = torch.abs(mask[key]) == len(checkpoint_paths)
        task_vector.vector[key] = torch.where(consistency_mask, merged_vector[key] / len(checkpoint_paths),
                                              torch.zeros_like(merged_vector[key]))

    args.eval_datasets = [dataset + "Val"]
    args.control_dataset = control_dataset + "Val"
    val_metrics = evaluate_task_vector(
        -task_vector,
        pretrained_checkpoint,
        args,
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric=f"{dataset}Val:top1",
        minimize=True,
        control_metric=f"{control_dataset}Val:top1",
        control_metric_threshold=0.95 * pretrained_accuracies[control_dataset + "Val"],
    )

    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset]
    args.control_dataset = control_dataset
    test_metrics = evaluate_task_vector_at_coef(
        -task_vector,
        pretrained_checkpoint,
        args,
        optimal_coef,
    )

    print("=" * 100)
    print(f"Test accuracy: {test_metrics[f'{dataset}:top1']}")

    negation_accuracies[dataset] = {
        "test": test_metrics[f"{dataset}:top1"],
        "test_control": test_metrics[f"{control_dataset}:top1"],
        "val": val_metrics,
    }

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/merge_result/negations_negmerge.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/merge_result/linear_negations_negmerge.json"

with open(save_file, "w") as f:
    json.dump(negation_accuracies, f, indent=4)
