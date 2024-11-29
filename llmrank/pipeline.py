# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from torch import nn
from transformers import PreTrainedTokenizerBase

from .model import LLMModel, MLPModel
from .score import BaseScorer
from .prune import BasePruner
from .utils import get_calibration_data

log = logging.getLogger()


class BasePipeline:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        structure_dict: dict,
        scorer: BaseScorer,
        pruner: BasePruner,
        device: str

    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.structure_dict = structure_dict
        self.scorer = scorer
        self.pruner = pruner
        self.device = device

    def run(self):
        self.model = LLMModel(
            self.model,
            self.structure_dict,
            device=self.device
        )
        dataloader = get_calibration_data(64, self.tokenizer, 64)
        self.model.extract_mlps()
        self.model.set_activations(dataloader)
        del dataloader
        self.model.model = self.model.model.cpu()
        self.pruning_loop()
        pruned_model = self.model.model

        del self.model
        return pruned_model

    def pruning_loop(self):
        raise NotImplementedError


class IsolatedFFNPipeline(BasePipeline):
    def pruning_loop(self):
        for i, mlp in enumerate(self.model.get_mlp_models()):
            self.scorer.iterate(mlp)
            self.pruner.iterate(mlp)
            log.debug(f"Layer (MLP) {i}")
        del mlp


class ChainedFFNPipeline(BasePipeline):
    def pruning_loop(self):
        mlp = MLPModel(self.model.get_layer_list())
        self.scorer.iterate(mlp)
        self.pruner.iterate(mlp)
        del mlp
