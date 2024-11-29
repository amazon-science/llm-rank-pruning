# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import datasets
import lm_eval
import torch
import transformers
from typing import Union


def model_size(model: torch.nn.Module) -> tuple[int, int, list]:
    """
    Counts the number of parameters in the model. Specifically it counts the
    total number of parameters as well as the number of non-zero parameters.
    """
    total_params = 0
    nonzero_params = 0
    param_ratios = []
    for param in model.parameters():
        t_curr = int(param.numel())
        nz_curr = int(torch.count_nonzero(param))
        total_params += t_curr
        nonzero_params += nz_curr
        param_ratios.append(nz_curr/t_curr)
    return total_params, nonzero_params, param_ratios


def path_load_dataset():
    """Monkey patch for trust remote code"""
    original_load_dataset = datasets.load_dataset

    def load_dataset_with_trust(*args, **kwargs):
        kwargs['trust_remote_code'] = True
        return original_load_dataset(*args, **kwargs)

    datasets.load_dataset = load_dataset_with_trust


def evaluate_llm(
    model: transformers.PreTrainedModel,
    tokenizer: Union[
        transformers.PreTrainedTokenizer,
        transformers.PreTrainedTokenizerFast,
    ],
    tasks: list[str],
    batch_size: Union[int, str] = "auto:8",
) -> dict:

    lm_model = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size
    )
    task_manager = lm_eval.tasks.TaskManager()
    path_load_dataset()
    results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=tasks,
        batch_size=batch_size,
        task_manager=task_manager
    )
    return results
