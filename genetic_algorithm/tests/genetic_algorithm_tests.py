# TESTING ==================================

## Imports ----------

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd

from genetic_algorithm.core import PipelineMaker, IndivMaker, GeneticAlgorithm

## Data ----------

df = pd.DataFrame({
    'x1': list(np.arange(1, 101)),
    'x2': [*['a']*50, *['b']*50]
})
df['y'] = (
    3 
    + 1*df['x1'] 
    + 4*(df['x2'] == 'b') 
    - 0.3*df['x1']*(df['x2'] == 'b') 
    + np.random.normal()
)
df.loc[5, 'x1'] = np.nan

## Pipeline test ----------

pipelineMaker = PipelineMaker(
    estimator_type='gbm_regressor', num_features=['x1'], cat_features=['x2'], random_state=617    
)            
pipe = pipelineMaker.make_pipeline(
    preprocessor_choices={
        'num_impute_strat': 'mean', 'cat_encoder_strat': 'one_hot', 'prior_frac': 0.1
    }, estimator_choices={
        'loss': 'ls', 'n_estimators': 100, 'subsample': 0.5, 
        'min_samples_leaf': 1, 'max_depth': 4, 'min_impurity_decrease': 0
    }
)

## Indiv maker test ----------

indivMaker = IndivMaker(
    estimator_type='gbm_regressor',  data_frame=df, target='y', 
    num_features=['x1'], cat_features=['x2'], cv_strat='KFold',
    n_splits=2, eval_criterion='neg_mean_squared_error', random_state=617
)
i1 = indivMaker.make_random_indiv()
i2 = indivMaker.make_random_indiv()
i3 = indivMaker.make_child_indiv(i1, i2)
indivMaker.mutate_indiv(i3)
indivMaker.assess_indiv_fitness(i1)

## Genetic algorithm test ----------

# class GeneticAlgorithm:

genAlg = GeneticAlgorithm(
    pop_size=20, top_frac=0.30, btm_frac=0.05, child_frac=0.25, mutate_frac=0.20,
    keep_graveyard=True, estimator_type='gbm_regressor', data_frame=df, target='y',
    num_features=['x1'], cat_features=['x2'], cv_strat='KFold', n_splits=2,
    eval_criterion='neg_mean_squared_error', random_state=617
)
genAlg._assess_population_fitness()
genAlg._kill_unfit()
genAlg._replenish_population()

genAlg.evolve(n_iters_max=2, print_current_best=True)
genAlg.evolve(n_iters_max=2, n_iters_no_improv_max=1, print_current_best=True)

if hasattr(genAlg, 'graveyard'):
    print(f'indivs in graveyard: {len(genAlg.graveyard)}')
    
print(f'best indiv: {genAlg.best_indiv}')

print('Done')