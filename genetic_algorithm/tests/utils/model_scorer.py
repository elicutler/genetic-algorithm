from genetic_algorithm.utils.model_scorer import ModelScorer
from genetic_algorithm.utils.pipeline_maker import PipelineMaker

import logging
from pprint import pprint 

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

n = 100
X = pd.DataFrame({'x1': np.random.normal(size=n)})
y = pd.Series()

pipelineMaker = PipelineMaker(
    estimatorClass=GradientBoostingRegressor,
    numFeatures=['x1', 'x2'], catFeatures=['g1', 'g2'],
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

evalMetric = 'mean_squared_error'
crossValidator = KFold(n_splits=3, shuffle=True, random_state=617)

modelScorer = ModelScorer(
    X=X, y=y, evalMetric='neg_mean_squared_error', 
    crossValidator=crossValidator, errorScore=np.nan
)

modelScorer.scoreModel(pipeline=pipeline, aggregator='mean')
