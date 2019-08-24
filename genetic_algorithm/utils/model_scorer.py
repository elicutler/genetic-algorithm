from typing import Union, Any

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from genetic_algorithm.utils.pipeline_maker import PipelineMaker

class ModelScorer:
    '''
    Class to evaluate model accuracy given data and evaluation criteria
    -----
    
    params
        estimator -- scikit-learn estimator or pipeline
        X -- input data
        y -- target data
        scoring -- metric to evaluate model accuracy
        crossValidator -- scikit-learn cross validation scheme
        errorScore -- how to score CV folds that encounter errors
        
    public methods
        scoreModel -- Evaluate model accuracy, given data and evaluation criteria
        
    public attributes
        X -- feature data
        y -- target data
        evalMetric -- evaluation metric for scoring
        crossValidator -- cross-validation strategy for scoring
    '''
    
    def __init__(
        self,
        X:Union[pd.DataFrame, np.array],
        y:[pd.Series, np.array],
        evalMetric:str,
        crossValidator:Any, # could be any number of classes from sklearn.model_selection
        errorScore:Union[float, int, str]=np.nan
    ):
        self.X = X
        self.y = y
        self.evalMetric = evalMetric
        self.crossValidator = crossValidator
        self.errorScore = errorScore
    
    def scoreModel(
        self, pipeline:Pipeline, aggregator:str='mean'
    ) -> float:
        '''
        score model using scikit-learn's cross_val_score
        -----
        
        params
            pipeline -- scikit-learn pipeline to score
            aggregator -- how to extract single metric from array of CV fold scores
            
        returns
            modelScore -- model score given data, eval metric, and cross-validator
        '''
        crossValScores = cross_val_score(
            estimator=pipeline, X=self.X, y=self.y, scoring=self.evalMetric,
            cv=self.crossValidator, error_score=self.errorScore
        )
        if aggregator == 'mean':
            modelScore = self._getMeanCrossValScore(crossValScores)
        return modelScore
    
    def _getMeanCrossValScore(self, crossValScores:np.array) -> float:
        meanCrossValScore = (
            crossValScores.mean() 
            if not np.isnan(crossValScores).all() else np.NINF
        )
        return meanCrossValScore
     