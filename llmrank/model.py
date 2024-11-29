# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from .utils import get_module


class LLMModel:
    def __init__(self, model, structure_dict, device) -> None:
        self.model = model
        self.device = torch.device(device)
        self.structure_dict = structure_dict
        self.mlp_models = []
        self.layer_list = []

    def extract_mlps(self) -> None:
        """
        This method extracts the mlp components of each decoder block and 
        appends it as an instance of the `MLPModel` Class to the `mlp_models`
        attribute. Additionally each mlp layer is appended to the `layer_list`
        attribute.
        """
        # Loop through LLM blocks
        layer_list = []
        flat_list = []
        llm_blocks = get_module(
            self.model,
            self.structure_dict["path_to_layers"]
        )
        for block_i in range(len(llm_blocks)):
            block_module = get_module(llm_blocks, [str(block_i)])
            layer_list = []
            for layer_path in self.structure_dict["path_to_modules"]:
                layer_module = get_module(block_module, layer_path)
                layer_list.append(layer_module)
                self.layer_list.append(layer_module)
            self.mlp_models.append(MLPModel(layer_list))
        return None

    def get_mlp_models(self):
        return self.mlp_models

    def get_layer_list(self):
        return self.layer_list

    def set_activations(self, dataloader):
        # Define the foward hook
        def hook_fn(module, input, output):
            out_activation = torch.norm(output, p=2, dim=(0, 1))
            in_activation = torch.norm(input[0], p=2, dim=(0, 1))
            assert len(out_activation) == int(module.weight.shape[0])
            assert len(in_activation) == int(module.weight.shape[1])
            if hasattr(module, "out_activation"):
                module.out_activation += out_activation
                module.in_activation += in_activation
                module.batch_counter += 1
            else:
                # Init cal activation aggregation
                module.out_activation = out_activation
                module.in_activation = in_activation
                module.batch_counter = 1
        # Register hooks with model
        hooks = []
        for layer in self.layer_list:
            hooks.append(layer.register_forward_hook(hook_fn))
        # Forward pass with calibration data
        for batch in dataloader:
            self.model(batch.to(self.device))
        # Remove hooks from the model
        for hook in hooks:
            hook.remove()
        # Compute average average across batches
        for layer in self.layer_list:
            layer.out_activation = layer.out_activation / layer.batch_counter
            layer.in_activation = layer.in_activation / layer.batch_counter
            del layer.batch_counter
        return None


class MLPModel:
    def __init__(self, layer_list) -> None:
        self.layer_list = layer_list

    def get_layer_list(self):
        return self.layer_list
