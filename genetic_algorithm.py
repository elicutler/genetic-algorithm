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

class PipelineBuilder:
    
    def __init__(self, estimator_type):
        self.preprocessor_choices = {
            'num_impute_strat': ['mean', 'median'],
            'cat_encoder_type': ['one_hot', 'target_mean'],
            'missing_values': [np.nan, None],
            'prior_frac': np.linspace(0.001, 1, num=100)
        } 

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
    

