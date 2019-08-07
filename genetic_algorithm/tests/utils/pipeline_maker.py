from genetic_algorithm.utils.pipeline_maker import PipelineMaker

import logging
from pprint import pprint 
from sklearn.ensemble import GradientBoostingRegressor

pipelineMaker = PipelineMaker(
    estimatorClass=GradientBoostingRegressor,
    numFeatures=['x1', 'x2'], catFeatures=['g1', 'g2'],
    randomState=617
)
pipe = pipelineMaker.makePipeline(
    preprocessorChoices={
        'numImputerStrat': 'median'
    },
    estimatorChoices={
        'learning_rate': 0.05, 'n_estimators': 50
    }
)
pprint(pipe.steps)
pprint(pipe.steps[0][1].transformer_list)

logging.info('All tests passed')
