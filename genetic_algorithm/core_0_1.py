import logging

import numpy as np

from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor, XGBClassifier

from genetic_algorithm.utils.sklearn_custom_transformers import TargetMeanEncoder
from genetic_algorithm.default_logger import DefaultLogger

class DefaultLogger: pass
defaultLogger = DefaultLogger()
defaultLogger.logger.setLevel(logging.INFO)

class GeneticAlgorithm:
    
    def __init__(
        self, 
        modelMaker: ModelMaker,
        modelScorer: ModelScorer,
        popSize:int,
        keepTopFrac:float,
        keepBtmFrac:float,
        makeChildFrac:float,
        mutateFrac:float,
        keepGraveyard:bool=False,
        randomState:int=None
    ) -> None:
        '''
        Genetic algorithm for scikit-learn model hyperparameter tuning 
        -----
        params
            :modelMaker: instance of class ModelMaker
            :modelScorer: instance of class ModelScorer
            :popSize: number of models in a generation
            :keepTopFrac: fraction of models in a generation to keep from top performers
            :keepBtmFrac: fraction of models in a generation to keeep from rest (randomized)
            :makeChildFrac: fraction of new models to spawn in a generation
            :mutateFrac: fraction of models to mutate in a generation
            :keepGraveyard: whether to keep a list of all trained models over generations
            :randomState: seed for random initializations
        public methods
            :evolve: evolve a population of models using genetic algorithm techniques
        '''
        assert keepTopFrac + keepBtmFrac + makeChildrenFrac <= 1        
        
        self.modelMaker = modelMaker
        self.modelScorer = modelScorer
        
        self.popSize = popSize
        self.keepTopFrac = keepTopFrac
        self.keepBtmFrac = keepBtmFrac
        self.makeChildFrac = makeChildFrac
        self.mutateFrac = mutateFrac
        self.keepGraveyard = keepGraveyard
        self.randomState = randomState
        
        self.keepTopN = int(np.floor(self.keepTopFrac * self.popSize))
        self.keepBtmN = int(np.floor(self.keepBtmFrac * self.popSize))
        self.makeChildN = int(np.floor(self.makeChildFrac * self.popSize))
        self.mutateN = int(np.floor(self.mutateFrac * self.popSize))
        
        if self.keepGraveyard:
            self.graveYard = []
            
        self.population = []
        self.bestModel = None   
        return None
            
    def evolve(
        self, 
        maxIters:int=10, 
        maxItersNoImprov:int=None, 
        printCurrentBest:bool=False
    ) -> None:
        '''
        Evolve the population until a stopping condition is met
        -----
        params:
            maxIters: maximum number of generations to evolve
            maxItersNoImprov: maximum number of consecutive 
                              generations to evolve without improvement
            printCurrentBest: log best loss after each evolution round
        void
        '''
        assert maxIters or maxItersNoImprov  # otherwise will run forever
        
        if len(self.population) == 0:
            self._initializePop()
            
        stopCond = False
        if maxIters is not None and itersNoImprov is not None:
            stopCond = iters == maxIters or itersNoImprov == maxItersNoImprov             
        elif maxIters is not None:
            stopCond = iters == maxIters
        elif itersNoImprov is not None:
            stopCond = itersNoImprov == maxItersNoImprov
        else:
            raise Exception('Invalid stopCond')
            
        iters = 0
        itersNoImprov = 0
        
        while not stopCond:

            self._scoreModelsInPop()
            bestModel = self._getBestModel()
            
            if self.bestModel is None or (
                bestModel.fitness > self.bestModel.fitness
            ):
                self.bestModel = bestModel
                itersNoImprov = 0
            else:
                itersNoImprov += 1
            iters += 1
            
            if printCurrentBest:
                print(f'Current best fitness: {self.bestModel.fitness}')
                
            self._killUnfit()
            self._makeChildren()
            self._makeRemainingRandomModels()
            
        print(
            f'Evolved {iters} generations ({itersNoImprov} generations '
            + 'without improvement)'
        )
        return None
    
    def _initializePop(self) -> None:
        assert len(self.population) == 0
        self.population = [
            self.modelMaker.makeRandomModel() for m in range(self.popSize)
        ]
        return None
        
    def _scoreModelsInPop(self) -> None:
        for m in range(self.popSize):
            if self.population[m].fitness is None:
                self.modelScorer.scoreModel(self.population[m])
        return None
    
    def _getBestModel(self) -> ModelMaker.model:
        self._sortPopByFitness()
        bestModel = self.population[0]
        return bestModel
        
    def _killUnfit(self) -> None:
        self._sortPopByFitness()

        topKeepMods = [self.population.pop(i) for i in range(self.keepTopN)]
        btmKeepInds = np.random.choice(range(len(self.population)), size=self.keepBtmN)
        btmKeepMods = [self.population.pop(i) for i in range(btmKeepInds)]
        
        if self.keepGraveyard:
            self.graveyard += self.population
        self.population = [*topKeepMods, *btmKeepMods]
        return None
        
    def _sortPopByFitness(self) -> None:
        self.population.sort(key=lambda m: m.fitness, reverse=True)
        return None
    
    def _makeChildren(self) -> None:
        children = []
        for i in range(self.makeChildN):
            mother, father = np.random.choice(self.population, size=2, replace=False)
            child = self.modelMaker.makeChildModel(mother, father)
            children.append(child)
        
        childToMutate = np.random.choice(children)
        self.modelMaker.mutateModel(childToMutate)
        
        self.population += children
        return None
    
    def _makeRemainingRandomModels(self) -> None:
        while len(self.population) < self.popSize:
            self.population.append(self.modelMaker.makeRandomModel())
        return None

        
class ModelMaker:
    # makeRandomModel()
    # makeChildModel()
    # mutateModel()
    def __init__(self) -> None:
        pass
    
    def makeRandomModel()
        
    
    
    preprocessorChoiceGrid = {
        'num_impute_strat': ['mean', 'median'],
        'cat_encoder_strat': ['one_hot', 'target_mean'],
        'missing_values': [np.nan, 'DO_NOT_FLAG_MISSING'],
        'prior_frac': np.linspace(0.01, 1, num=100)
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




class ModelScorer:
    pass