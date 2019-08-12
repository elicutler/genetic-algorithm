from typing import Optional, Dict

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)

from genetic_algorithm.utils.pipeline_maker import PipelineMaker

class ModelMaker:
    '''
    Class to make scikit-learn pipeline models
    -----
    params
        :pipelineMaker: scikit-learn pipeline maker
        :estimatorType: supervised estimator (maps to scikit-learn estimator class)
        :preprocessorChoiceGridOverrides: optional preprocessor choice grids 
                                          to override defaults
        :estimatorChoiceGridOverrides: optional estimator choice grids 
                                       to override defaults
    public methods
        :makeRandomModel: Makes random model based on choice grids
        :makeChildModel: Makes model by randomly combining hyperparameters
                         of two models
        :mutateModel: Mutate a model by randomly changing one of its hyperparameters
    '''
    def __init__(
        self, 
        pipelineMaker: PipelineMaker,
        preprocessorChoiceGridOverrides: Optional[Dict[str, list]] = None,
        estimatorChoiceGridOverrides: Optional[Dict[str, list]] = None,
    ):
        self.pipelineMaker = pipelineMaker
        self.preprocessorChoiceGridOverrides = preprocessorChoiceGridOverrides
        self.estimatorChoiceGridOverrides = estimatorChoiceGridOverrides
        
        if self.pipelineMaker.estimatorClass == GradientBoostingRegressor:
            self.estimatorChoiceGrid = self.gbmRegressorChoiceGrid
        elif self.pipelineMaker.estimatorClass == RandomForestRegressor:
            self.estimatorChoiceGrid = self.rfRegressorChoiceGrid
        elif self.pipelineMaker.estimatorClass == ElasticNet:
            self.estimatorChoiceGrid = self.enetRegressorChoiceGrid
        elif self.pipelineMaker.estimatorClass == GradientBoostingClassifier:
            self.estimatorChoiceGrid = self.gbmClassifierChoiceGrid
        elif self.pipelineMaker.estimatorClass == RandomForestClassifier:
            self.estimatorChoiceGrid = self.rfClassifierChoiceGrid
        elif self.pipelineMaker.estimatorClass == SGDClassifier:
            self.estimatorChoiceGrid = self.enetClassifierChoiceGrid
            
        if self.preprocessorChoiceGridOverrides is not None:
            self.preprocessorChoiceGrid = {
                **self.preprocessorChoiceGrid, 
                **self.preprocessorChoiceGridOverrides
            }
        if self.estimatorChoiceGridOverrides is not None:
            self.estimatorChoiceGrid = {
                **self.estimatorChoiceGrid,
                **self.estimatorChoiceGridOverrides
            }
    
    def makeRandomModel(self):
        preprocessorChoices = {
            param: np.random.choice(self.preprocessorChoiceGrid[param])
            for param in self.preprocessorChoiceGrid.keys()
        }
        estimatorChoices = {
            param: np.random.choice(self.estimatorChoiceGrid[param])
            for param in self.estimatorChoiceGrid.keys()
        }
        randomModel = self.pipelineMaker.makePipeline(
            preprocessorChoices, estimatorChoices
        )
        return randomModel
    
    def makeChildModel(self, mother, father):
        preprocessorChoices = {
            param: np.random.choice([
                mother.preprocessorChoices[param],
                father.preprocessorChoices[param]
            ]) for param in self.preprocessorChoiceGrid.keys()
        }
        estimatorChoices = {
            param: np.random.choice([
                mother.estimatorChoices[param],
                father.estimatorChoices[param]
            ]) for param in self.estimatorChoiceGrid.keys()
        }
        childModel = self.pipelineMaker.makePipeline(
            preprocessorChoices, estimatorChoices
        )
        return childModel
    
    def mutateModel(self, childModel: Pipeline) -> None:
        
        paramType = np.random.choice(['preprocessor', 'estimator'])
        
        if paramType == 'preprocessor':
            params = childModel.preprocessorChoices
            choiceGrid = self.preprocessorChoiceGrid
        elif paramType == 'estimator':
            params = childModel.estimatorChoices
            choiceGrid = self.estimatorChoiceGrid
            
        paramToMutate = np.random.choice(params.keys())
        paramValue = params[paramToMutate]
        paramChoices = choiceGrid[paramToMutate]
        breakpoint()
        
        
        
        
    preprocessorChoiceGrid = {
        'numImputerStrat': ['mean', 'median'],
        'catEncoderStrat': ['oneHot', 'targetMean'],
        'missingValues': [np.nan, 'DO_NOT_FLAG_MISSING'],
        'tmePriorFrac': np.linspace(0.01, 1, num=100)
    }
    gbmRegressorChoiceGrid = {
        'loss': ['ls', 'lad'],
        'n_estimators': np.arange(100, 1000, 100),
        'subsample': np.linspace(0.1, 1, num=10),
        'min_samples_leaf': np.arange(1, 10),
        'max_depth': np.arange(1, 12),
        'min_impurity_decrease': np.linspace(0, 1, num=10)
    }
    rfRegressorChoiceGrid = {
        'criterion': ['mse', 'mae'],
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': np.arange(100, 1000, 100)
    }
    enetRegressorChoiceGrid = {
        # sklearn advises against including very small alpha values
        'alpha': np.linspace(0.01, 1, num=100), 
        'l1_ratio': np.concatenate(
            [np.logspace(-3, -1, num=4), np.linspace(0, 1, num=100)]
        )
    }
    gbmClassifierChoiceGrid = {
        'learning_rate': np.linspace(0.01, 1, num=100),
        'n_estimators': np.arange(100, 1000, 100),
        'subsample': np.linspace(0.1, 1, num=10),
        'min_samples_leaf': np.arange(2, 10),
        'max_depth': np.arange(1, 12),
        'min_impurity_decrease': np.linspace(0, 1, num=10)        
    }
    rfClassifierChoiceGrid = {
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': np.arange(100, 1000, 100)
    }
    enetClassifierChoiceGrid = {
        'loss': ['hinge', 'log'],
        'alpha': np.concatenate(
            [np.logspace(-4, -2, num=3), np.linspace(0.1, 1, num=100)]
        ),
        'l1_ratio': np.concatenate(
            [np.logspace(-4, -1, num=4), np.linspace(0, 1, num=100)]
        ),
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': np.concatenate(
            [np.logspace(-4, -2, num=3), np.linspace(0.1, 1, num=100)]
        ),
        'power_t': np.concatenate(
            [np.logspace(-4, -2, num=3), np.linspace(0.1, 1, num=100)]
        ),
        'penalty': ['elastic_net']
    }
