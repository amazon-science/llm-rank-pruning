# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import math
import torch
import torch.nn.utils.prune as prune


class BasePruner:
    def __init__(self, amount: float) -> None:
        self.amount = amount

    def iterate(self, model) -> None:
        layer_list = model.get_layer_list()
        self.extract_scores(layer_list)
        for i, layer in enumerate(model.get_layer_list()):
            self.prune(layer)
            self.clean_up(layer)

    def prune(self, layer) -> None:
        if hasattr(layer, "amount"):
            amount = layer.amount
        else:
            amount = self.amount
        # Create pruning buffer
        prune.ln_structured(
            module=layer,
            name="weight",
            importance_scores=layer.pruning_mask,
            amount=amount,
            dim=0,
            n=1,
        )
        # Make pruning permanent
        prune.remove(layer, "weight")

    def scores_to_matrix(self, score_vector, expand_dim):
        matrix = (
            score_vector
            .unsqueeze(1)
            .repeat(1, expand_dim)
        )
        return matrix

    def extract_scores(self, layer_list):
        raise Exception("Not implemented, use child class")

    def clean_up(self, layer):
        if hasattr(layer, "amount"):
            del layer.amount
        del layer.in_activation
        del layer.out_activation
        del layer.pruning_mask
        del layer.importance
        gc.collect()
        torch.cuda.empty_cache()


class LocalPruner(BasePruner):
    def __init__(self, amount: float) -> None:
        super().__init__(amount)

    def extract_scores(self, layer_list):
        for layer in layer_list:
            layer.pruning_mask = super().scores_to_matrix(
                layer.importance,
                layer.in_features
            )


class GlobalPruner(BasePruner):
    def __init__(self, amount: float) -> None:
        super().__init__(amount)

    def extract_scores(self, layer_list):
        flat_scores = torch.cat(
            [
                layer.importance
                for layer in layer_list
                if hasattr(layer, "weight")
            ],
            dim=0,
        )
        threshold_idx = math.ceil(len(flat_scores) * self.amount)
        lowest_nodes = flat_scores.argsort()[:threshold_idx]
        flat_mask = torch.ones(len(flat_scores))
        for i in lowest_nodes:
            flat_mask[i] = 0
        # Reshape to pruning masks
        pointer = 0
        for layer in layer_list:
            # if not hasattr(layer["module"], "weight"):
            #     continue
            layer_mask = flat_mask[
                pointer: pointer + layer.out_features
            ]
            pointer += layer.out_features
            layer.pruning_mask = super().scores_to_matrix(
                layer.importance,
                layer.in_features
            )
            layer.amount = int(
                layer_mask.numel() - torch.count_nonzero(layer_mask)
            )
