from typing import List, Dict, Any, Optional, Union

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

class PipelineMaker:
    '''
    Class for making scikit-learn pipelines with extra attributes for GeneticAlgorithm
    -----
    
    params
        estimatorClass -- scikit-learn ML estimator class
        numFeatures -- numeric features for model
        catFeatures -- categorical features for model
        randomState -- seed for initializing estimator
        
    public methods
        makePipeline -- make scikit-learn pipeline, with some extra attributes
        
    public attributes
        estimatorClass -- type of scikit-learn estimator
        numFeatures -- numeric features
        catFeatures -- categorical features
        randomState -- seed for initializing estimator
    '''
    def __init__(
        self,
        estimatorClass:Any, # could be any number of sklearn classes
        numFeatures:List[str],
        catFeatures:List[str],
        randomState:Optional[int]=None
    ) -> None:
        self.estimatorClass = estimatorClass
        self.numFeatures = numFeatures
        self.catFeatures = catFeatures
        self.randomState = randomState
    
    def makePipeline(
        self,
        preprocessorChoices:Dict[str, List[Any]],
        estimatorChoices:Dict[str, List[Any]]
    ) -> Pipeline:
        '''
        Makes scikit-learn pipeline, with some extra attributes
        -----
        
        params
            preprocessorChoices -- choice grid for each preprocessor hyperparameter
            estimatorChoices --choice grid for each estimator hyperparameter

        returns
            pipeline -- enhanced scikit-learn pipeline
        '''
        
        preprocessor = self._makePreprocessor(**preprocessorChoices)
        estimator = self._makeEstimator(estimatorChoices)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('estimator', estimator)
        ])
        pipeline.preprocessorChoices = preprocessorChoices
        pipeline.estimatorChoices = estimatorChoices
        pipeline.fitness = None
        return pipeline
    
    def _makePreprocessor(
        self, 
        numImputerStrat:str='mean',
        catEncoderStrat:str='oneHot',
        missingValues:Union[float, str]=np.nan,
        tmePriorFrac:Optional[float]=None
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
        catEncoderStrat:str, 
        tmePriorFrac:Optional[float]=None
    ) -> Union[OneHotEncoder, TargetMeanEncoder]:
        
        if catEncoderStrat == 'oneHot':
            catEncoder = OneHotEncoder(handle_unknown='ignore')
        elif catEncoderStrat == 'targetMean':
            catEncoder = TargetMeanEncoder(priorFrac=tmePriorFrac)
        return catEncoder
        
    def _makeEstimator(
        self, 
        estimatorChoices:dict,
    ) -> Any: 
        estimator = self.estimatorClass(
            **estimatorChoices, random_state=self.randomState
        )
        return estimator