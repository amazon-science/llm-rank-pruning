# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from accelerate import dispatch_model, infer_auto_device_map
from datetime import datetime, timedelta
import json
import gc
import logging
import os
import torch


from llmrank.utils import load_hf_model
from llmrank.config import MODEL_CONFIGS, SCORERS, PRUNERS

from experiments.evaluation import model_size, evaluate_llm
from experiments.config import EVAL_TASKS, REPRESENTAION_MAP


def batch_experiment(
    model_names: list[str],
    scorer_names: list[str],
    pruner_names: list[str],
    amounts: list[float],
    tasks: list[str],
    wandb_logs: bool,
    local_path: str,
    device: str,
    pipeline: str
) -> dict:
    if wandb_logs:
        import wandb
    batch_results = dict()
    for model_name in model_names:
        for score_name in scorer_names:
            for pruner_name in pruner_names:
                for amount in amounts:
                    experiment_name = (
                        f"{model_name}_{score_name}_{pruner_name}_{amount}_{pipeline}"
                    )
                    logging.info(f"Start Experiment: {experiment_name}")
                    experiment_config = {
                        "model_name": model_name,
                        "score_name": score_name,
                        "pruner_name": pruner_name,
                        "pipeline": pipeline,
                        "device": device,
                        "amount": amount,
                        "tasks": tasks
                    }
                    experiment_result = single_experiment(**experiment_config)
                    if wandb_logs:
                        wandb.init(
                            project="LLM-Rank-Pruning",
                            name=experiment_name,
                            config=experiment_config
                        )
                        wandb.log(experiment_result)
                        wandb.finish()
                    if local_path:
                        os.makedirs(local_path, exist_ok=True)
                        result_file_name = experiment_name + ".json"
                        result_path = os.path.join(
                            local_path, result_file_name)
                        with open(result_path, "w") as file:
                            json.dump(experiment_config, file)
    return batch_results


def single_experiment(
    model_name: str,
    score_name: str,
    pruner_name: str,
    amount: float,
    pipeline: str,
    device: str,
    tasks: list[str]
) -> dict:
    config = MODEL_CONFIGS[model_name]
    scorer = SCORERS[score_name]()
    pruner = PRUNERS[pruner_name](amount=amount)
    model, tokenizer = load_hf_model(
        config["local_path"], config["hf_path"], "auto")
    device_map = model.hf_device_map
    pipeline = REPRESENTAION_MAP[pipeline](
        model, tokenizer, config["structure_dict"], scorer, pruner, device
    )
    if amount != 0:
        # Prune Model
        pruning_start_time = datetime.now()
        pipeline.run()
        pruning_runtime = datetime.now() - pruning_start_time
    else:
        pruning_runtime = timedelta(seconds=0)
    # Evaluate model size
    tl_params, nz_params, param_ratios = model_size(model)
    # Load model to cuda
    # device_map = infer_auto_device_map(model)
    model = dispatch_model(model, device_map=device_map)
    # Benchmark model
    eval_start_time = datetime.now()
    benchmarks = evaluate_llm(model, tokenizer, tasks)
    eval_runtime = datetime.now() - eval_start_time
    result_dict = {
        "tl_params": tl_params,
        "nz_params": nz_params,
        "param_ratios": param_ratios,
        "benchmarks": benchmarks['results'],
        "pruning_walltime": pruning_runtime.total_seconds(),
        "eval_walltime": eval_runtime.total_seconds()
    }
    # Clean up
    model = model.cpu()
    del scorer
    del pruner
    del model
    del tokenizer
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_max_memory_allocated(i)
        torch.cuda.reset_max_memory_cached(i)
    gc.collect()
    torch.cuda.empty_cache()

    return result_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-names", type=str, nargs="+", default=MODEL_CONFIGS.keys())
    parser.add_argument(
        "--scorer-names", type=str, nargs="+", default=SCORERS.keys())
    parser.add_argument(
        "--pruner-names", type=str, nargs="+", default=PRUNERS.keys())
    parser.add_argument(
        "--amounts", type=float, nargs="+", default=[0.3])
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=EVAL_TASKS)
    parser.add_argument(
        "--wandb-logs", action="store_true")
    parser.add_argument(
        "--local-path", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda")
    parser.add_argument(
        "--pipeline", type=str, choices=["SingleFFN", "ChainedFFN"], default="SingleFFN")

    return parser.parse_args()


if __name__ == "__main__":
    # Get CLI arguments
    args = get_args()
    logging.info((
        f"Start experiment batch for:\n"
        f" -> models: {args.model_names};\n"
        f" -> scorers: {args.scorer_names};\n"
        f" -> pruners: {args.pruner_names};\n"
        f" -> amounts: {args.amounts};\n"
        f" -> tasks: {args.tasks};\n"
        f" -> device: {args.device};\n"
        f" -> pipeline: {args.pipeline};\n"
    ))
    # Run experiments
    batch_start_time = datetime.now()
    results = batch_experiment(
        model_names=args.model_names,
        scorer_names=args.scorer_names,
        pruner_names=args.pruner_names,
        amounts=args.amounts,
        tasks=args.tasks,
        wandb_logs=args.wandb_logs,
        local_path=args.local_path,
        device=args.device,
        pipeline=args.pipeline
    )
    batch_runtime = datetime.now() - batch_start_time
    logging.info(f"Experiment batch finished after {batch_runtime}!")
