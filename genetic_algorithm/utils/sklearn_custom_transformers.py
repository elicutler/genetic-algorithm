from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        priorSize: Optional[int] = None, 
        priorFrac: Optional[float] = None
    ):
        assert not (priorSize is not None and priorFrac is not None), (
            'Optionally set either priorSize or priorFrac, but not both'
        )        
        self.priorSize = priorSize
        self.priorFrac = priorFrac
        
    def fit(
        self, 
        X: Union[pd.DataFrame, np.array], 
        y: Union[pd.Series, np.array]
    ) -> TargetMeanEncoder:
        
        X = self._makeSameDimArray(X)
        y = self._makeNx1Array(y)

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
    
    @staticmethod
    def _getSameDimArray(X: Union[pd.DataFrame, np.array]) -> np.array:
        sameDimArray = X.values if isinstance(X, pd.DataFrame) else X  
        return xArray
    
    @staticmethod
    def _makeNx1Array(y: Union[pd.Series, np.array]) -> np.array:
        if isinstance(y, pd.Series):
            Nx1Array = y.values.reshape(y.shape[0], 1)  
        elif np.ndim(y) == 1:
            Nx1Array = y.reshape(y.shape[0], 1)
        return Nx1Array
        
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