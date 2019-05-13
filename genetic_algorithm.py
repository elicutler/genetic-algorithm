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

from sklearn_custom_transformers import TargetMeanEncoder

# Pipeline building blocks ==========

class PipelineMaker:
    
    def __init__(self, estimator_type, random_state=617):
          
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
                'min_samples_leaf': np.arange(2, 10),
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
        self.estimator_type = estimator_type
        self.random_state = random_state
        
    def _make_preprocessor(
        self, num_features, cat_features, num_impute_strat,
        cat_encoder_strat, missing_values, prior_frac
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
            ('num_pipe', num_pipe, num_features),
            ('cat_pipe', cat_pipe, cat_features)
        ])        
        preprocessor = FeatureUnion([
            ('num_cat_pipe', num_cat_transformers),
            ('missing_flagger', MissingIndicator(missing_values=missing_values, features='all'))
        ])    
        return preprocessor
    
    def make_pipeline(self, num_features, cat_features, preprocessor_choices, estimator_choices):
        
        preprocessor = self._make_preprocessor(num_features, cat_features, **preprocessor_choices)
        
        if self.estimator_type == 'gbm_regressor':
            estimator = GradientBoostingRegressor(**estimator_choices)
        elif self.estimator_type == 'gbm_classifier':
            estimator = GradientBoostingClassifier(**estimator_choices)
            
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('estimator', estimator)
        ])
        return pipeline
            
        

        
        

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
    

