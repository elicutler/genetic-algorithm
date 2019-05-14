'''
Genetic algorithm for ML hyperparameter tuning
'''

# Imports ==========

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

from utils.sklearn_custom_transformers import TargetMeanEncoder

# Pipeline building blocks ==========

class PipelineMaker:
    
    def __init__(self, estimator_type, num_features, cat_features, random_state=617):
        
        self.estimator_type = estimator_type
        self.num_features = num_features
        self.cat_features = cat_features
        self.random_state = random_state
        
    def _make_preprocessor(
        self, num_impute_strat, cat_encoder_strat, missing_values, prior_frac
    ):           
        if cat_encoder_strat == 'one_hot':
            cat_encoder = OneHotEncoder(handle_unknown='ignore')
        elif cat_encoder_strat == 'target_mean':
            cat_encoder = TargetMeanEncoder(prior_frac=prior_frac)  
            
        num_pipe = Pipeline([
            ('num_imputer', SimpleImputer(strategy=num_impute_strat)),
            ('num_normalizer', StandardScaler())
        ])         
        cat_pipe = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('cat_encoder', cat_encoder)
        ])        
        num_cat_pipe = ColumnTransformer([
            ('num_pipe', num_pipe, self.num_features),
            ('cat_pipe', cat_pipe, self.cat_features)
        ])        
        preprocessor = FeatureUnion([
            ('num_cat_pipe', num_cat_pipe),
            ('missing_flagger', MissingIndicator(missing_values=missing_values, features='all'))
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
            
            
pipelineMaker = PipelineMaker(
    estimator_type='gbm_regressor', num_features=['x1', 'x2'], cat_features=['c1', 'c2']
)            
z = pipelineMaker.make_pipeline(
    preprocessor_choices={
        'num_impute_strat': 'mean', 'cat_encoder_strat': 'one_hot', 
        'missing_values': np.nan, 'prior_frac': 0.1
    }, estimator_choices={
        'loss': 'ls', 'n_estimators': 100, 'subsample': 0.5, 
        'min_samples_leaf': 1, 'max_depth': 4, 'min_impurity_decrease': 0
    }
)        


class IndivMaker:
    
    def __init__(
        self, estimator_type, num_features, cat_features, random_state=617
    ):
        self.pipelineMaker = PipelineMaker(
            estimator_type=estimator_type, num_features=num_features, 
            cat_features=cat_features, random_state=random_state
        )
        self.preprocessor_choice_grid = {
            'num_impute_strat': ['mean', 'median'],
            'cat_encoder_strat': ['one_hot', 'target_mean'],
            'missing_values': [np.nan, None],
            'prior_frac': np.linspace(0.01, 1, num=100)
        }
        if estimator_type == 'gbm_regressor':
            self.estimator_choice_grid = {
                'loss': ['ls', 'lad'],
                'n_estimators': np.arange(100, 1000, 100),
                'subsample': np.linspace(0.1, 1, num=10),
                'min_samples_leaf': np.arange(1, 10),
                'max_depth': np.arange(12),
                'min_impurity_decrease': np.linspace(0, 1, num=10)
            }
        elif estimator_type == 'gbm_classifier':
            self.estimator_choice_grid = {
                'learning_rate': np.linspace(0.01, 1, num=100),
                'n_estimators': np.arange(100, 1000, 100),
                'subsample': np.linspace(0.1, 1, num=10),
                'min_samples_leaf': np.arange(2, 10),
                'max_depth': np.arange(12),
                'min_impurity_decrease': np.linspace(0, 1, num=10)        
            }
            
    def _make_indiv(self, preprocessor_choice_grid, estimator_choice_grid):
        preprocessor_choices = {
            param: np.random.choice(preprocessor_choice_grid[param])
            for param in preprocessor_choice_grid.keys()
        }
        estimator_choices = {
            param: np.random.choice(estimator_choice_grid[param])
            for param in estimator_choice_grid.keys()
        }
        indiv = self.pipelineMaker.make_pipeline(
            preprocessor_choices=preprocessor_choices, estimator_choices=estimator_choices
        )
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
            for param in mother['preprocessor_choices'].keys()
        }
        indiv = self._make_indiv(
            preprocessor_choice_grid=preprocessor_choice_grid,
            estimator_choice_grid=estimator_choice_grid
        )
        return indiv
    
    def mutate_indiv(self, indiv):
        pipeline_choice_grid = {**self.preprocessor_choice_grid, **self.estimator_choice_grid}
        gene_to_mutate = np.random.choice(list(pipeline_choice_grid.keys()))
        assert len(pipeline_choice_grid[gene_to_mutate]) > 1
        
        mutation = np.random.choice(pipeline_choice_grid[gene_to_mutate])
        if indiv[]

# Genetic algorithm ==========

class GeneticAlgorithm:
    
    def __init__(self):
        pass
    
    def _replenish_population(self):
        pass
    
    def _assess_population_fitness(self):
        pass
    
    def _kill_unfit(self):
        pass
    
    def _evolve_generation(self):
        pass
    
    def evolve(self, n_iters=10, n_iters_no_improvement=None):
        pass
    

