'''
Contains custom transformers for scikit-learn
'''

# Imports ==========

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

# Create target mean encoder ==========

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, prior_size=0, prior_frac=0):
        
        assert not (prior_size > 0 and prior_frac > 0)
        
        self.prior_size = prior_size
        self.prior_frac = prior_frac
        
    def fit(self, X, y):
        
        X = X.values if isinstance(X, pd.DataFrame) else X  

        if isinstance(y, pd.Series):
            y = y.values.reshape(y.shape[0], 1)  
        elif np.ndim(y) == 1:
            y = y.reshape(y.shape[0], 1)

        data_arr = np.concatenate((X, y), axis=1)
        
        grand_mean = np.mean(y)
        
        lvl_means  = {col: {} for col in range(X.shape[1])}
        lvl_counts = {col: {} for col in range(X.shape[1])}

        for col in lvl_means.keys():
            for lvl in np.unique(X[:, col]):

                X_col_lvl = data_arr[:, col] == lvl
                y_col     = data_arr.shape[1] - 1   
                
                X_col_lvl_y_numeric = data_arr[X_col_lvl, y_col].astype(float)
                
                lvl_means[col][lvl]  = (
                    np.mean(X_col_lvl_y_numeric) 
                    if not np.isnan(np.mean(X_col_lvl_y_numeric))
                    else grand_mean
                )
                
                if np.isnan(lvl_means[col][lvl]):
                    breakpoint()
                
                lvl_counts[col][lvl] = data_arr[X_col_lvl].shape[0]
                
        self.lvl_means = lvl_means
                
        if self.prior_size or self.prior_frac:
            
            lvl_means_smoothed = {col: {} for col in range(X.shape[1])}
            
            for col in lvl_means.keys():
                for lvl in lvl_means[col].keys():
                    
                    if self.prior_size:
                        smooth_wt = self.prior_size / (self.prior_size + lvl_counts[col][lvl])
                    elif self.prior_frac:
                        prior_size = X.shape[0]*self.prior_frac
                        smooth_wt  = prior_size / (prior_size + lvl_counts[col][lvl])
                        
                    lvl_means_smoothed[col][lvl] = (
                        (1 - smooth_wt)*lvl_means[col][lvl] + smooth_wt*grand_mean
                        if not np.isnan((1 - smooth_wt)*lvl_means[col][lvl] + smooth_wt*grand_mean)
                        else grand_mean
                    )
                    
                    if np.isnan(lvl_means_smoothed[col][lvl]):
                        breakpoint()
                        
            self.lvl_means_smoothed = lvl_means_smoothed
            self.grand_mean = grand_mean
                
        return self
        
    def transform(self, X):
        
        X = X.values if isinstance(X, pd.DataFrame) else X
        X_transformed = np.empty_like(X)
        
        for col in range(X_transformed.shape[1]):
            
            if self.prior_size or self.prior_frac:
                means_getter = np.vectorize(self.lvl_means_smoothed[col].get)
            else:
                means_getter = np.vectorize(self.lvl_means[col].get)
                
            X_transformed[:, col] = means_getter(X[:, col])
                
        return X_transformed