from typing import List, Optional, Union

import numpy as np

from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from genetic_algorithm.utils.sklearn_custom_transformers import TargetMeanEncoder
# from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
# from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# from xgboost import XGBRegressor, XGBClassifier


class PipelineMaker:
    def __init__(
        self,
        estimatorClass,
        numFeatures: List[str],
        catFeatures: List[str],
        randomState: Optional[int] = None
    ):
        self.estimatorClass = estimatorClass
        self.numFeatures = numFeatures
        self.catFeatures = catFeatures
        self.randomState = randomState
        return None
    
    def makePipeline(
        self,
        preprocessorChoices: list,
        estimatorChoices: list
    ) -> Pipeline:
        
        preprocessor = self._makePreprocessor(**preprocessorChoices)
        estimator = self._makeEstimator(estimatorChoices)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('estimator', estimator)
        ])
        return pipeline
    
    def _makePreprocessor(
        self, 
        numImputerStrat: str = 'mean',
        catEncoderStrat: str = 'oneHot',
        missingValues: Union[float, str] = np.nan,
        tmePriorFrac: Optional[float] = None
    ) -> FeatureUnion:
        
        catEncoder = self._getCatEncoder(catEncoderStrat)
        numPipe = Pipeline([
            ('numImputer', SimpleImputer(strategy=numImputerStrat)),
            ('numScaler', StandardScaler())
        ])
        catPipe = Pipeline([
            ('catImputer', SimpleImputer(strategy='most_frequent')),
            ('catEncoder', catEncoder)
        ])
        numCatPipe = ColumnTransformer([
            ('numPipe', numPipe, self.numFeatures),
            ('catPipe', catPipe, self.catFeatures)
        ])
        preprocessor = FeatureUnion([
            ('numCatPipe', numCatPipe),
            ('missingFlagger', 
             MissingIndicator(missing_values=missingValues, features='all')
            )
        ])
        return preprocessor
        
    @staticmethod
    def _getCatEncoder(
        catEncoderStrat: str, 
        tmePriorFrac: Optional[float] = None
    ) -> Union[OneHotEncoder, TargetMeanEncoder]:
        
        if catEncoderStrat == 'oneHot':
            catEncoder = OneHotEncoder(handle_unknown='ignore')
        elif catEncoderStrat == 'targetMean':
            catEncoder = TargetMeanEncoder(prior_frac=tmePriorFrac)
        return catEncoder
        
    def _makeEstimator(
        self, 
        estimatorChoices: dict,
    ): # make conditional based on estimator class
        estimator = self.estimatorClass(
            **estimatorChoices, random_state=self.randomState
        )
        return estimator