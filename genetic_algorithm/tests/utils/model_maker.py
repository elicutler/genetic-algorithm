from genetic_algorithm.utils.model_maker import ModelMaker
from genetic_algorithm.utils.pipeline_maker import PipelineMaker

import logging
from pprint import pprint

from sklearn.ensemble import GradientBoostingRegressor

pipelineMaker = PipelineMaker(
    estimatorClass=GradientBoostingRegressor,
    numFeatures=['x1', 'x2'], catFeatures=['g1', 'g2'],
    randomState=617
)

modelMaker = ModelMaker(pipelineMaker=pipelineMaker)

mod1 = modelMaker.makeRandomModel()
mod2 = modelMaker.makeRandomModel()
mod1 == mod2

modelMaker = ModelMaker(
    pipelineMaker=pipelineMaker,
    preprocessorChoiceGridOverrides={
      'numImputerStrat': ['hot-diggidy-dog', 'shamboozle']  
    },
    estimatorChoiceGridOverrides={
        'n_estimators': [20, 30]
    }
)
mod3 = modelMaker.makeRandomModel()
pprint(mod3)
pprint(mod3.steps[0][1].transformer_list)

logging.info('All tests passed')
