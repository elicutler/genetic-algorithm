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
        returns
            self
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
            self._replenishPop()
            
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
    
    def _
        
        
                
                
        
        