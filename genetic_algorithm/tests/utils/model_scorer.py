from genetic_algorithm.utils.model_scorer import ModelScorer
from genetic_algorithm.utils.pipeline_maker import PipelineMaker

import logging
from pprint import pprint 

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

X = pd.DataFrame({'g1': ['a', 'a', 'b', 'b'], 'g2': ['c', 'd', 'd', 'd']})
y = pd.Series([1, 5, 7, 9])

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

evalMetric = 'mean_squared_error'
crossValidator = KFold(n_splits=3, shuffle=True, random_state=617)

modelScorer = ModelScorer(
    X=X, y=y, evalMetric='mean_squared_error', 
    crossValidator=crossValidator, errorScore=np.nan
)
