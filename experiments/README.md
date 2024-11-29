# Benchmarking Experiments

The [experiments-module](experiments.py) offers a CLI tool for running batched benchmarking with the llmrank package. 

## How to Use

| Arguments    | Description                                                                                       | Options                                                                            | Default                                                                            |
|--------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| model-names  | Takes a space separated list of model names registered in the [MODEL_CONFIGS](../llmrank/config.py).  | "open_llama_3b_v2"                                                                 | "open_llama_3b_v2"                                                                 |
| scorer-names | Takes a space separated list of scorer names registered in the [SCORERS](../llmrank/config.py) config.       | "norm", "random", "activation", "wpr", "mpr", "wanda"                              | "norm", "random", "activation", "wpr", "mpr", "wanda"                              |
| pruner-names | Takes a space separated list of pruner names registered in the [PRUNERS](../llmrank/config.py) config.       | "local", "global"                                                                  | "local", "global"                                                                  |
| amounts      | Takes a space separated list of pruning amounts.                                                  | $$\text{Number} \in [0,1)$$                                                        | [0.3]                                                                              |
| tasks        | A space separated list of benchmarking tasks which is fed to the lm evaluation harness framework. | "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "wikitext" | "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "wikitext" |
| wandb-logs   | A flag that enables logging via wandb if it is set.                                               | None (set the flag or not)                                                         | False                                                                              |
| local-path   | If this is set, the experiment results will be saved to this path in JSON format.                 | "valid/local/path/"                                                                | None                                                                               |
| device       | Takes a valid device string to pass to `torch.device()` which is used for pruning the model.      | "cuda", "cpu"                                                                      | "cuda"                                                                             |
| pipeline     | Takes a pipeline name registered in the [REPRESENTAION_MAP](config.py) config.                            | "SingleFFN", "ChainedFFN"                                                          | "SingleFFN"                                                                        |

## Examples

To run all scoring function with 30% sparsity for FFNs with the chained pipeline, use this command:
```
python -m experiments.experiments \
    --model-names open_llama_3b_v2 \
    --scorer-names norm random activation wpr mpr \
    --pruner-names local \
    --amounts 0.3 \
    --wandb-logs \
    --pipeline ChainedFFN
```