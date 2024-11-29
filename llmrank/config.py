# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .prune import LocalPruner, GlobalPruner
from .score import (
    NormScorer, RandomScorer, ActivationScorer,
    WeightedPageRankScorer, ModifiedPageRankScorer, WandaScorer
)

MODEL_CONFIGS = {
    "open_llama_3b_v2": {
        "local_path": "./artifacts/model/open_llama_3b_v2/",
        "hf_path": "openlm-research/open_llama_3b_v2",
        "structure_dict": {
            "path_to_layers": ["model", "layers"],
            "path_to_modules": (
                ("mlp", "up_proj"),
                ("mlp", "down_proj")
            )
        }
    }
}
SCORERS = {
    "norm": NormScorer,
    "random": RandomScorer,
    "activation": ActivationScorer,
    "wpr": WeightedPageRankScorer,
    "mpr": ModifiedPageRankScorer,
    "wanda": WandaScorer
}

PRUNERS = {
    "local": LocalPruner,
    "global": GlobalPruner
}
