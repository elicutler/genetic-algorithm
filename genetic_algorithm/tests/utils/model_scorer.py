from genetic_algorithm.utils.model_scorer import ModelScorer
from genetic_algorithm.utils.pipeline_maker import PipelineMaker

import logging
from pprint import pprint 

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

n = 100
X = pd.DataFrame({
    'x1': np.random.normal(size=n), 
    'g1': np.random.choice(['a', 'b', 'c'], size=n)
})
y = pd.Series(
    0.5 + 0.5*X['x1'] + 1.0*(X['g1'] == 'b') + 1.5*(X['g1'] == 'c') 
    + np.random.normal(size=n)
)

pipelineMaker = PipelineMaker(
    estimatorClass=GradientBoostingRegressor,
    numFeatures=['x1'], catFeatures=['g1'],
    randomState=617
)
pipeline = pipelineMaker.makePipeline(
    preprocessorChoices={
        'numImputerStrat': 'median'
    },
    estimatorChoices={
        'learning_rate': 0.05, 'n_estimators': 50
    }
)

evalMetric = 'neg_mean_squared_error'
crossValidator = KFold(n_splits=3, shuffle=True, random_state=617)

modelScorer = ModelScorer(
    X=X, y=y, evalMetric=evalMetric, 
    crossValidator=crossValidator, errorScore=np.nan
)

modelScorer.scoreModel(pipeline=pipeline, aggregator='mean')

logging.info('All tests passed')