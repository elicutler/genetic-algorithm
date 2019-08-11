from genetic_algorithm.core import GeneticAlgorithm
from genetic_algorithm.utils.model_maker import ModelMaker
from genetic_algorithm.utils.model_scorer import ModelScorer
from genetic_algorithm.utils.pipeline_maker import PipelineMaker

from genetic_algorithm.utils.default_logger import DefaultLogger

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

n = 100
X = pd.DataFrame({
    'x1': np.random.normal(size=n), 
    'g1': np.random.choice(['a', 'b', 'c'], size=n)
})
y = pd.Series(
    0.5 + 0.5*X['x1'] + 1.0*(X['g1'] == 'b') + 1.5*(X['g1'] == 'c') 
    + np.random.normal(size=n)
)