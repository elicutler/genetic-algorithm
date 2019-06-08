'''
Genetic algorithm for ML hyperparameter tuning
'''

# Imports ==========

import numpy as np

from sklearn.impute import MissingIndicator
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor, XGBClassifier

from genetic_algorithm.utils.sklearn_custom_transformers import TargetMeanEncoder

# Pipeline building blocks ==========

class PipelineMaker:
    
    def __init__(self, estimator_type, num_features, cat_features, random_state=None):
        
        self.estimator_type = estimator_type
        self.num_features = num_features
        self.cat_features = cat_features
        self.random_state = random_state
        
    def _make_preprocessor(
        self, num_impute_strat, cat_encoder_strat, prior_frac
    ):           
        if cat_encoder_strat == 'one_hot':
            cat_encoder = OneHotEncoder(handle_unknown='ignore')
        elif cat_encoder_strat == 'target_mean':
            cat_encoder = TargetMeanEncoder(prior_frac=prior_frac)  
            
        num_pipe = Pipeline([
            ('num_imputer', Imputer(strategy=num_impute_strat)),
            ('num_normalizer', StandardScaler())
        ])         
        cat_pipe = Pipeline([
            ('cat_imputer', Imputer(strategy='most_frequent')),
            ('cat_encoder', cat_encoder)
        ])        
        num_cat_pipe = ColumnTransformer([
            ('num_pipe', num_pipe, self.num_features),
            ('cat_pipe', cat_pipe, self.cat_features)
        ])        
        preprocessor = FeatureUnion([
            ('num_cat_pipe', num_cat_pipe),
            ('missing_flagger', MissingIndicator(features='all'))
        ])    
        return preprocessor
    
    def make_pipeline(self, preprocessor_choices, estimator_choices):
        
        preprocessor = self._make_preprocessor(**preprocessor_choices)
        
        if self.estimator_type == 'gbm_regressor':
            estimator = GradientBoostingRegressor(random_state=self.random_state, **estimator_choices)
        elif self.estimator_type == 'gbm_classifier':
            estimator = GradientBoostingClassifier(random_state=self.random_state, **estimator_choices)
            
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('estimator', estimator)
        ])
        return pipeline
                 

class IndivMaker:
    
    def __init__(
        self, estimator_type, data_frame, target, num_features, cat_features,  
        cv_strat, n_splits, eval_criterion, random_state=None
    ):
        self.estimator_type = estimator_type
        self.data_frame = data_frame
        self.target = target
        self.num_features = num_features
        self.cat_features = cat_features
        self.cv_strat = cv_strat
        self.n_splits = n_splits
        self.eval_criterion = eval_criterion
        self.random_state = random_state
        
        self.pipelineMaker = PipelineMaker(
            estimator_type=self.estimator_type, num_features=self.num_features, 
            cat_features=self.cat_features, random_state=self.random_state
        )
        self.preprocessor_choice_grid = {
            'num_impute_strat': ['mean', 'median'],
            'cat_encoder_strat': ['one_hot', 'target_mean'],
            'prior_frac': np.linspace(0.01, 1, num=100)
        }
        if self.estimator_type == 'gbm_regressor':
            self.estimator_choice_grid = {
                'loss': ['ls', 'lad'],
                'n_estimators': np.arange(100, 1000, 100),
                'subsample': np.linspace(0.1, 1, num=10),
                'min_samples_leaf': np.arange(1, 10),
                'max_depth': np.arange(1, 12),
                'min_impurity_decrease': np.linspace(0, 1, num=10)
            }
        elif self.estimator_type == 'gbm_classifier':
            self.estimator_choice_grid = {
                'learning_rate': np.linspace(0.01, 1, num=100),
                'n_estimators': np.arange(100, 1000, 100),
                'subsample': np.linspace(0.1, 1, num=10),
                'min_samples_leaf': np.arange(2, 10),
                'max_depth': np.arange(1, 12),
                'min_impurity_decrease': np.linspace(0, 1, num=10)        
            }
        if self.cv_strat == 'KFold':
            self.cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv == 'StratifiedKFold':
            self.cv = StratifiedKfold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv == 'TimeSeriesSplit':
            self.cv = TimeSeriesSplit(n_splits=self.n_splits)        
            
    def _make_indiv(self, preprocessor_choice_grid, estimator_choice_grid):
        preprocessor_choices = {
            param: np.random.choice(preprocessor_choice_grid[param])
            for param in preprocessor_choice_grid.keys()
        }
        estimator_choices = {
            param: np.random.choice(estimator_choice_grid[param])
            for param in estimator_choice_grid.keys()
        }
        pipe = self.pipelineMaker.make_pipeline(
            preprocessor_choices=preprocessor_choices, estimator_choices=estimator_choices
        )
        indiv = {
            'preprocessor_choices': preprocessor_choices,
            'estimator_choices': estimator_choices,
            'pipe': pipe
        }
        return indiv
    
    def make_random_indiv(self):
        indiv = self._make_indiv(
            preprocessor_choice_grid=self.preprocessor_choice_grid,
            estimator_choice_grid=self.estimator_choice_grid
        )
        return indiv
    
    def make_child_indiv(self, mother, father):
        preprocessor_choice_grid = {
            param: [
                mother['preprocessor_choices'][param], 
                father['preprocessor_choices'][param]
            ]
            for param in mother['preprocessor_choices'].keys()
        }
        estimator_choice_grid = {
            param: [
                mother['estimator_choices'][param],
                father['estimator_choices'][param]
            ]
            for param in mother['estimator_choices'].keys()
        }
        indiv = self._make_indiv(
            preprocessor_choice_grid=preprocessor_choice_grid,
            estimator_choice_grid=estimator_choice_grid
        )
        return indiv
    
    def mutate_indiv(self, indiv):        
        preprocessor_gene = np.random.choice(list(self.preprocessor_choice_grid.keys()))
        preprocessor_mutation = np.random.choice(self.preprocessor_choice_grid[preprocessor_gene])
        indiv['preprocessor_choices'][preprocessor_gene] = preprocessor_mutation
        
        estimator_gene = np.random.choice(list(self.estimator_choice_grid.keys()))
        estimator_mutation = np.random.choice(self.estimator_choice_grid[estimator_gene])
        indiv['estimator_choices'][estimator_gene] = estimator_mutation
        
        return None
    
    def assess_indiv_fitness(self, indiv):
        pipe = indiv['pipe']
        X = self.data_frame[[*self.num_features, *self.cat_features]]
        y = np.ravel(self.data_frame[self.target]) # RandomForest raises warning if not passed 1D array
        cv_scores = cross_val_score(
            estimator=pipe, X=X, y=y, scoring=self.eval_criterion, cv=self.cv, error_score=np.nan
        )
        indiv['fitness'] = cv_scores.mean() if not np.isnan(cv_scores.mean()) else np.NINF
        return None
        

# Genetic algorithm ==========

class GeneticAlgorithm:
    
    def __init__(
        self, pop_size, top_frac, btm_frac, child_frac, mutate_frac, keep_graveyard, 
        estimator_type, data_frame, target, num_features, cat_features, 
        cv_strat, n_splits, eval_criterion, random_state=None
    ):
        assert top_frac + btm_frac + child_frac <= 1
        
        self.pop_size = pop_size
        self.top_n    = int(np.floor(top_frac*self.pop_size))
        self.btm_n    = int(np.floor(btm_frac*self.pop_size))
        self.child_n  = int(np.floor(child_frac*self.pop_size))
        self.mutate_n = int(np.floor(mutate_frac*self.child_n))
        
        self.keep_graveyard = keep_graveyard        
        if self.keep_graveyard:
            self.graveyard = []
            
        self.estimator_type = estimator_type
        self.data_frame = data_frame
        self.target = target
        self.num_features = num_features
        self.cat_features = cat_features
        self.cv_strat = cv_strat
        self.n_splits = n_splits
        self.eval_criterion = eval_criterion
        self.random_state = random_state
        
        self.indivMaker = IndivMaker(
            estimator_type=self.estimator_type, data_frame=self.data_frame, target=self.target,
            num_features=self.num_features, cat_features=self.cat_features, cv_strat=self.cv_strat, 
            eval_criterion=self.eval_criterion, n_splits=self.n_splits, random_state=self.random_state
        )
        
        self.population = [
            self.indivMaker.make_random_indiv()
            for i in range(self.pop_size)
        ]        
        self.best_indiv = None
        self.n_iters_total = 0
    
    def _assess_population_fitness(self):
        for indiv in range(len(self.population)):
            if 'fitness' not in self.population[indiv].keys():
                self.indivMaker.assess_indiv_fitness(self.population[indiv])
        return None

    def _kill_unfit(self):        
        pop_indexes = list(range(self.pop_size))
        top_n_indexes = pop_indexes[:self.top_n]
        remaining_indexes = [index for index in pop_indexes if index not in top_n_indexes]
        btm_n_indexes = np.random.choice(remaining_indexes, size=self.btm_n)
        kill_indexes = [index for index in remaining_indexes if index not in btm_n_indexes]
        keep_indexes = [*top_n_indexes, *btm_n_indexes]
        
        if self.keep_graveyard:
            for index in kill_indexes:
                self.graveyard += [self.population[index]]
                
        population = [self.population[index] for index in keep_indexes]
        self.population = population
        return None
                
    def _replenish_population(self):
        assert len(self.population) >= 2
        
        children = []
        for i in range(self.child_n):
            mother = np.random.choice(self.population)
            father = np.random.choice(self.population)
            while father == mother:
                father = np.random.choice(self.population)
            child_indiv = self.indivMaker.make_child_indiv(mother, father)
            children += [child_indiv]
            
        for i in range(self.mutate_n):
            indiv_to_mutate = np.random.choice(children)
            self.indivMaker.mutate_indiv(indiv_to_mutate)
            
        self.population += children
        
        while len(self.population) < self.pop_size:
            self.population += [self.indivMaker.make_random_indiv()]
    
        return None
    
    def evolve(self, n_iters_max=10, n_iters_no_improv_max=None, print_current_best=False):
        
        assert n_iters_max or n_iters_no_improv_max
        
        n_iters = 0
        n_iters_no_improv = 0
        
        while True:
            self._assess_population_fitness()
            self.population.sort(key=lambda indiv: indiv['fitness'], reverse=True)
            best_indiv_current_gen = self.population[0]
            
            n_iters += 1
            self.n_iters_total += 1
            
            if self.best_indiv is None or (
                best_indiv_current_gen['fitness'] > self.best_indiv['fitness']
            ):
                self.best_indiv = best_indiv_current_gen
                n_iters_no_improv = 0
            else:
                n_iters_no_improv += 1  
                
            if print_current_best:
                print('Current best fitness: {}'.format(self.best_indiv['fitness']))
            
            if (n_iters_max and n_iters >= n_iters_max) or (
                n_iters_no_improv_max and n_iters_no_improv >= n_iters_no_improv_max
            ):
                print(
                    f'Finished evolving {n_iters} generations '
                    + f'({n_iters_no_improv} consecutive generations without improvement)'
                )
                break
                
            self._kill_unfit()
            self._replenish_population()           
        
        return None

