
# LLM-Rank

This is the official PyTorch implementation for pruning Large Language models using the LLM-Rank pruning method. It is based on the weighted PageRank centrality measure as introduced in our paper:

> **LLM-Rank: A Graph Theoretical Approach to Pruning Large Language Models** \
> Amazon Web Services - AI Research \
> Author: David B. Hoffmann \
> Advisor: Dr. Kailash Budhathoki, Dr. Matthaeus Kleindessner \
> Paper: https://arxiv.org/abs/2410.13299

```bibtex
@article{hoffmann2024llmrank,
    title={LLM-Rank: A Graph Theoretical Approach to Pruning Large Language Models},
    author={Hoffmann, David B. and Budhathoki, Kailash and Kleindessner, Matthaeus},
    year={2024},
    journal={arXiv preprint arXiv:2410.13299}
}
```

## Setup

1. Clone the repository with `git@github.com:amazon-science/llm-rank-pruning.git/`
2. Navigate into the repository with `cd llm-rank-pruning`
3. Install the `llmrank` module with `pip install -e .`

## How To Use

The package provides everything needed to perform post training pruning of large language models. The **llmrank** package contains the actual pruning code of the LLM-Rank method as well as the other baselines we benchmark in our paper. It is designed to be easily extendible and compatible with further methods and scoring functions for structured pruning to allow for other comparisons. The extension to new models and methods is described in the [custom extensions](llmrank/README.md#include-other-pruning-methods) section.

The entry point to the llmrank package is the pipeline module which provides different pipelines for pruning and is documented in the [llmrank README](llmrank/README.md#how-to-use). To prune the feed forward layers of an open_llama_3b_v2 model using weighted PageRank centrality with C4 as calibration data, the following code can be used: 

```python
from llmrank import ChainedFFNPipeline
from llmrank.score import WeightedPageRankScorer
from llmrank.prune import LocalPruner
from llmrank.utils import load_hf_model

# Load model
model, tokenizer = load_hf_model("./artifacts/model/open_llama_3b_v2/") 

# Define model structure
structure_dict = {
    "path_to_layers": ["model", "layers"],
    "path_to_modules": ( 
        ("mlp", "up_proj"),
        ("mlp", "down_proj")
    )
}
scorer = WeightedPageRankScorer()
pruner = LocalPruner(amount=0.3)

pipeline = ChainedFFNPipeline(model, tokenizer, structure_dict, scorer, pruner, "cuda")
pruned_model = pipeline.run()
```
The **experiment** module acts as a CLI tool for running batched experiments. It takes a list of fixed and iterable arguments for different networks, scoring functions, pruning methods and pruning amounts and creates and runs the relevant pipeline for each experiment. For more details refer to the [experiment README](experiments/README.md). 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
