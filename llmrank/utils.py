# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset, load_from_disk
import transformers
from torch.utils.data import Dataset, DataLoader
import random
import os


def close_to_one(n, tol: float = 0.0001) -> bool:
    """
    Checks if a number is close to one with respect to the tol parameter.
    """
    return abs(n - 1) < tol


def get_module(module, path_list):
    new_module = getattr(module, path_list[0])
    if len(path_list) > 1:
        return get_module(new_module, path_list[1:])
    else:
        return new_module


class CalibrationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_calibration_data(nsamples, tokenizer, batch_size=16):
    # Load train and validation datasets
    local_data_path = "./artifacts/data/allenai_c4.hf"
    exists_locally = os.path.exists(local_data_path)
    if exists_locally:
        calibration_data = load_from_disk(local_data_path)
    else:
        calibration_data = load_dataset(
            'allenai/c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train'
        )
        # Save data locally
        calibration_data.save_to_disk(local_data_path)
    # Prepare dataset for dataloader
    seqlen = tokenizer.model_max_length
    # Generate samples from training set
    random.seed(42)
    samples = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(calibration_data) - 1)
            # Will raise a warning for too long sequences (truncated later)
            trainenc = tokenizer(
                calibration_data[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert len(inp) <= seqlen
        samples.append(inp[0])
    dataset = CalibrationDataset(samples)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader


def load_hf_model(
    local_path: str = "./artifacts/model/open_llama_3b_v2",
    hf_path: str = "openlm-research/open_llama_3b_v2",
    device: str = "auto"
) -> tuple:
    exists_locally = os.path.exists(local_path)
    if exists_locally:
        model_path = local_path
    else:
        model_path = hf_path
    # Load model
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_path,
        device_map=device,
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_path,
        legacy=True,
    )
    if not exists_locally:
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
    return model, tokenizer
