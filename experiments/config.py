# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from llmrank import ChainedFFNPipeline, IsolatedFFNPipeline

EVAL_TASKS = [
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "wikitext"
]

REPRESENTAION_MAP = {
    "SingleFFN": IsolatedFFNPipeline,
    "ChainedFFN": ChainedFFNPipeline
}
