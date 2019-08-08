from typing import Union

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from genetic_algorithm.utils.pipeline_maker import PipelineMaker

class ModelScorer:
    '''
    Class to evaluate model accuracy given data and evaluation criteria
    -----
    params
        :estimator: scikit-learn estimator or pipeline
        :X: input data array
        :y: target data array
        :scoring: metric to evaluate model accuracy
        :crossValidator: scikit-learn cross validation scheme
        :errorScore: how to score CV folds that encounter errors
    public methods
        :scoreModel: Evaluate model accuracy, given data and evaluation criteria
    '''
    def __init__(
        self,
        X: np.array,
        y: np.array,
        evalMetric,
        crossValidator,
        errorScore: Union[float, int, str] = np.nan
    ):
        self.X = X
        self.y = y
        self.evalMetric = evalMetric
        self.crossValidator = crossValidator
        self.errorScore = errorScore
    
    def scoreModel(
        self, 
        pipeline: Pipeline,
        aggregator: str = 'mean'
    ) -> float:
        '''
        score model using scikit-learn's cross_val_score
        -----
        params
            :aggregator: how to extract single metric from array of CV fold scores
        returns
            model score (float)
        '''
        crossValScores = cross_val_score(
            estimator=pipeline, X=self.X, y=self.y, scoring=self.evalMetric,
            cv=self.crossValidator, error_score=self.errorScore
        )
        if aggregator == 'mean':
            modelScore = self._getMeanCrossValScore(crossValScores)
        return modelScore
    
    def _getMeanCrossValScore(self, crossValScores: np.array) -> float:
        meanCrossValScore = (
            crossValScores.mean() 
            if not np.isnan(crossValScores).all() else np.NINF
        )
        return meanCrossValScore
     