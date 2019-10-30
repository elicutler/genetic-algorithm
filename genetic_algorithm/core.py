from typing import Optional
import logging

import numpy as np
from sklearn.pipeline import Pipeline

from genetic_algorithm.utils.model_maker import ModelMaker
from genetic_algorithm.utils.model_scorer import ModelScorer

class Population:
    '''
    Genetic algorithm for scikit-learn model hyperparameter tuning 
    -----
    
    params
        modelMaker -- instance of class ModelMaker
        modelScorer -- instance of class ModelScorer
        popSize -- number of models in a generation
        keepTopFrac -- fraction of models in a generation to keep from top performers
        keepBtmFrac -- fraction of models in a generation to keep from rest (randomized)
        makeChildFrac -- fraction of new models to spawn in a generation
        mutateFrac -- fraction of models to mutate in a generation
        keepGraveyard -- whether to keep a list of all trained models over generations
        randomState -- seed for random initializations
        
    public methods
        evolve -- evolve a population of models using genetic algorithm techniques
        
    public attributes
        self.population -- final population of models
        self.bestModel -- best model after all iterations
        self.totalGensEvolved -- total number of generations evolved
        self.graveYard -- discarded models from previous generations (if kept)
    '''
    def __init__(
        self, 
        modelMaker:ModelMaker,
        modelScorer:ModelScorer,
        popSize:int,
        keepTopFrac:float,
        keepBtmFrac:float,
        makeChildFrac:float,
        mutateFrac:float,
        keepGraveyard:bool=False,
        randomState:Optional[int]=None
    ) -> None:
        assert keepTopFrac + keepBtmFrac + makeChildFrac <= 1        
        
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
            self.graveyard = []
            
        self.population = []
        self.bestModel = None  
        self.totalGensEvolved = 0
        
        if self.randomState is not None:
            np.random.seed(self.randomState)
            
    def evolve(
        self, 
        maxIters:Optional[int]=10, 
        maxItersNoImprov:Optional[int]=None, 
        logCurrentBest:bool=False
    ) -> None:
        '''
        Evolve the population until a stopping condition is met
        -----
        params
            maxIters -- maximum number of generations to evolve
            maxItersNoImprov -- maximum number of consecutive 
                generations to evolve without improvement
            printCurrentBest -- log best loss after each evolution round
            
        void
        '''
        assert maxIters is not None or maxItersNoImprov is not None 
        
        if len(self.population) == 0:
            self._initializePop()
            
        iters = 0
        itersNoImprov = 0
        stopEvolving = False
        
        while not stopEvolving:
            self._scoreModelsInPop()
            bestModelCurGen = self._getBestModelCurGen()
            
            if self.bestModel is None or (
                bestModelCurGen.fitness > self.bestModel.fitness
            ):
                self.bestModel = bestModelCurGen
                itersNoImprov = 0
            else:
                itersNoImprov += 1
            iters += 1
            self.totalGensEvolved += 1
            
            if logCurrentBest:
                logging.info(f'Current best fitness: {self.bestModel.fitness}')
                
            self._killUnfit()
            self._makeChildren()
            self._makeRemainingRandomModels()
            
            stopEvolving = self._checkIfStopCondMet(
                iters, itersNoImprov, maxIters, maxItersNoImprov
            )
            
        logging.info(
            f'Evolved {iters} generations ({itersNoImprov} generations '
            + 'without improvement)'
        )
    
    def _initializePop(self) -> None:
        assert len(self.population) == 0, 'Models already in population'
        self.population = [
            self.modelMaker.makeRandomModel() for m in range(self.popSize)
        ]
        
    def _scoreModelsInPop(self) -> None:
        for m in range(len(self.population)):
            if self.population[m].fitness is None:
                self.population[m].fitness = (
                    self.modelScorer.scoreModel(self.population[m])
                )
    
    def _getBestModelCurGen(self) -> Pipeline:
        self._sortPopByFitness()
        bestModelCurGen = self.population[:1].copy()[0]
        return bestModelCurGen
        
    def _killUnfit(self) -> None:
        self._sortPopByFitness()

        topKeepInds = range(self.keepTopN)
        topKeepMods = [self.population[i] for i in topKeepInds]
        
        remainingInds = [i for i in range(len(self.population)) if i not in topKeepInds]
        btmKeepInds = np.random.choice(
            remainingInds, size=self.keepBtmN, replace=False
        )
        btmKeepMods = [self.population[i] for i in btmKeepInds]

        assert len(set(topKeepInds).intersection(set(btmKeepInds))) == 0, (
            'Cannot have overlap in top kept models and bottom kept models'
        )
        
        keepInds = [*topKeepInds, *btmKeepInds]
        keepMods = [self.population[i] for i in keepInds]
        
        if self.keepGraveyard:
            buryMods = [
                self.population[i] for i in range(len(self.population)) 
                if i not in keepInds
            ]
            self.graveyard += buryMods
            
        self.population = keepMods
        
    def _sortPopByFitness(self) -> None:
        self.population.sort(key=lambda m: m.fitness, reverse=True)
    
    def _makeChildren(self) -> None:
        children = []
        for i in range(self.makeChildN):
            mother, father = np.random.choice(self.population, size=2, replace=False)
            child = self.modelMaker.makeChildModel(mother, father)
            children.append(child)
        
        childrenToMutate = np.random.choice(children, size=self.mutateN, replace=False)
        for child in childrenToMutate:
            self.modelMaker.mutateModel(child)
        
        self.population += children
    
    def _makeRemainingRandomModels(self) -> None:
        while len(self.population) < self.popSize:
            self.population.append(self.modelMaker.makeRandomModel())
    
    @staticmethod
    def _checkIfStopCondMet(
        iters:int, 
        itersNoImprov:int,
        maxIters:Optional[int], 
        maxItersNoImprov:Optional[int]
    ) -> bool:    
        if maxIters is not None and itersNoImprov is not None:
            stopEvolving = iters == maxIters or itersNoImprov == maxItersNoImprov             
        elif maxIters is not None:
            stopEvolving = iters == maxIters
        elif itersNoImprov is not None:
            stopEvolving = itersNoImprov == maxItersNoImprov
        else:
            raise Exception(f'Invalid stopEvolving value: {stopEvolving}')
        return stopEvolving
