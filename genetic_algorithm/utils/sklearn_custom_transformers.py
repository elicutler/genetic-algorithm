from typing import Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        priorSize: Optional[int] = None, 
        priorFrac: Optional[float] = None
    ):
        
        assert not (priorSize > 0 and priorFrac > 0)
        
        self.priorSize = priorSize
        self.priorFrac = priorFrac
        
    def fit(self, X, y):
        
        X = X.values if isinstance(X, pd.DataFrame) else X  

        if isinstance(y, pd.Series):
            y = y.values.reshape(y.shape[0], 1)  
        elif np.ndim(y) == 1:
            y = y.reshape(y.shape[0], 1)

        dataArr = np.concatenate((X, y), axis=1)
        
        grandMean = np.mean(y)
        
        levelMeans = {j: {} for j in range(X.shape[1])}
        levelCounts = {j: {} for j in range(X.shape[1])}

        for j in levelMeans.keys():
            for g in np.unique(X[:, j]):

                X_jg = dataArr[:, j] == g
                y_j = dataArr.shape[1] - 1   
                
                X_jg_y = data_arr[X_jg, y_j].astype(float)
                
                lvl_means[j][g]  = (
                    np.mean(X_jg_y) 
                    if not np.isnan(np.mean(X_jg_y))
                    else grand_mean
                )
                
                lvl_counts[j][g] = dataArr[X_jg].shape[0]
                
        self.levelMeans = levelMeans
                
        if self.prior_size > 0 or self.prior_frac > 0:
            
            levelMeansSmoothed = {j: {} for j in range(X.shape[1])}
            
            for j in levelMeans.keys():
                for g in levelMeans[j].keys():
                    
                    if self.priorSize > 0:
                        weightSmoothed = self.priorSize / (self.priorSize + levelCounts[j][g])
                    elif self.priorFrac > 0:
                        priorSize = X.shape[0]*self.priorFrac
                        weightSmoothed = priorSize / (priorSize + levelCounts[j][g])
                        
                    levelMeansSmoothed_jg = (
                        (1 - weightSmoothed)*levelMeans[j][g] + weightSmoothed*grandMean
                    )                        
                    levelMeansSmoothed[j][g] = (
                        levelMeansSmoothed_jg
                        if not np.isnan(levelMeansSmoothed_jg)
                        else grand_mean
                    )
                        
            self.levelMeansSmoothed = levelMeansSmoothed
            self.grandMean = grandMean
                
        return self
        
    def transform(self, X: np.array) -> np.array:
        
        X = X.values if isinstance(X, pd.DataFrame) else X
        XTransformed = np.empty_like(X).astype(float)
        
        for j in range(XTransformed.shape[1]):
            
            if self.priorSize > 0 or self.priorFrac > 0:
                meansGetter = np.vectorize(self.levelMeansSmoothed[j].get)
            else:
                meansGetter = np.vectorize(self.levelmeans[j].get)
                
            XTransformed[:, j] = meansGetter(X[:, j])
            
            # imputes for levels in validation set not seen in training set
            XTransformed[np.isnan(XTransformed[:, j]), j] = self.grandMean
                
        return XTransformed