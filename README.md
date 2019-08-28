# Usage pattern

```
import genetic_algorithm as G

pipelineMaker = G.PipelineMaker(...)
modelMaker = G.ModelMaker(pipelineMaker=pipelineMaker, ...)
modelScorer = G.modelScorer(...)

genAlg = G.GeneticAlgorithm(modelMaker=modelMaker, modelScorer=modelScorer, ...)
genAlg.evolve(...)
genAlg.bestModel.fit(...)

genAlg.bestModel.predict(...)
