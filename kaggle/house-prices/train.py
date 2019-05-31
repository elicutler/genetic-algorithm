# Imports ==========

import sys
import os
sys.path.insert(0, os.path.abspath('../../genetic_algorithm'))

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from genetic_algorithm import GeneticAlgorithm

# Read in data ==========

train_df = pd.read_csv('data/train.csv')
test_df  = pd.read_csv('data/test.csv')

# Process data ==========

data_frames = [train_df, test_df]

def process_data(in_df):
    col_id = in_df['Id']
    df = in_df.drop(columns=['Id', 'MiscFeature', 'MiscVal'])
    return col_id, df

train_id, train_df = process_data(train_df)
test_id, test_df   = process_data(test_df)

num_features = [col for col in train_df.columns if is_numeric_dtype(train_df[col]) and col != target]
cat_features = [col for col in train_df.columns if col not in num_features and col != target]

train_df['logSalePrice'] = np.log(train_df['SalePrice'])

target = 'logSalePrice'

# Train models ==========

genAlg = GeneticAlgorithm(
    pop_size=20, top_frac=0.30, btm_frac=0.05, child_frac=0.25, mutate_frac=0.20,
    keep_graveyard=True, estimator_type='gbm_regressor', data_frame=train_df, target=target,
    num_features=num_features, cat_features=cat_features, cv_strat='KFold', n_splits=2,
    eval_criterion='neg_mean_squared_error', random_state=617
)

genAlg.evolve(n_iters_max=5, n_iters_no_improv_max=3, print_current_best=True)

if hasattr(genAlg, 'graveyard'):
    print(f'indivs in graveyard: {len(genAlg.graveyard)}')
    
print(f'best indiv: {genAlg.best_indiv}')

print('Done')

# Prepare final predictions for submission ==========

genAlg.best_indiv['fitness']
genAlg.best_indiv['pipe'].fit(train_df[[*num_features, *cat_features]], train_df[target])
test_preds = np.exp(
    genAlg.best_indiv['pipe'].predict(test_df[[*num_features, *cat_features]])
)
test_preds_final = pd.DataFrame({'Id': test_id, 'SalePrice': test_preds})

# Write final submissions to CSV ==========

now = pd.Timestamp.now().strftime('%Y-%m-d %H:%M:%S')
test_preds_final.to_csv(f'out/test_preds_final {now}.csv', index=False)

