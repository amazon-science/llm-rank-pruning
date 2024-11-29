# LLMRank Pruning

This README provides an overview over the different pruning pipelines, pruning methods and scoring methods included in this implementation. Furthermore, it provides pointers on how the package can be extended by other pruning and scoring methods. 

## Pipelines

The pruning pipelines defined in [pipeline.py](pipeline.py) perform end-to-end pruning of a pretrained LLM. Two types of pruning are offered as extensions of the BasePipeline:
- **IsolatedFFNPipeline**: Treats each position wise FFN in the LLM as an individual multilayer perceptron and applies mlp-rank-pruning to it. This is the easiest conceivable graph representation but may not use the full potential of weighted PageRank pruning. 
- **ChainedFFNPipeline**: Connect all FFNs to a single large multilayer perceptron and applies mlp-rank-pruning to it. This is motivated by the fact that most information flows through the skip connection and individual component only add small changes, which means that treating all FFNs as connected may be a viable simplification. 

## Pruning Methods
This package refers to methods of removing nodes based on scores that have already been computed as pruning methods. It contains two method by default as part of the [prune module](prune.py).
- **LocalPruner**: In local pruning each already scored layer of a neural network is considered individually. The pruner then removes the elements with the lowest score from each layer. In the case of structured pruning the same share of nodes (e.g. 30%) is pruned for each layer. 
- **GlobalPruner**: With global pruning all layers are considered and the globally lowest scoring nodes are removed. This can lead to a different amount of nodes being pruned in different layers. 

## Scoring Functions:

| Scorer                 | Description                                                                                                                                               |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| NormScorer             | Computes the channel p norm importance scores for a module of the network, given the weight matrix of the module.                                         |
| ActivationScorer       | Computes the random importance scores for a module of the network, given the out_activation of the module.                                                |
| RandomScorer           | Computes the random importance scores for a module of the network.                                                                                        |
| WeightedPageRankScorer | Computes the page-rank score for a module of the network, given the weight matrix of the module and the importance score of the previous module.          |
| ModifiedPageRankScorer | Computes the modified page-rank score for a module of the network, given the weight matrix of the module and the importance score of the previous module. |
| WandaScorer            | Computes the agregated channel wanda scores for a module of the network, given the out_activation of the module.                                          |

## Extend Pruning & Scoring

The llmrank is designed such that additional scoring and pruning methods can easily be added as extensions of the respective base classes. To include an additional scoring function simply implement a child class of `BaseScorer` with a score method which returns a vector of importance scores for a single network layer. 

To implement a new pruning method simple create a child class of `BasePruner` which implements the extract_scores method that takes a list of neural network modules and adds a pruning_mask tensor to each module and amount based on which the network will be pruned. 