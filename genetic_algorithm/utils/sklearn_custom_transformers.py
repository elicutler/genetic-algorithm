from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    '''
    Target mean encoding data preprocessor compatible with scikit-learn pipelines
    -----
    params
        :priorSize: 
    '''
    
    def __init__(
        self,
        priorSize: Optional[int] = None, 
        priorFrac: Optional[float] = None
    ):
        assert not (priorSize is not None and priorFrac is not None)
        
        self.priorSize = priorSize
        self.priorFrac = priorFrac
        
    def fit(
        self, 
        X: Union[pd.DataFrame, np.array], 
        y: Union[pd.Series, np.array]
    ) -> 'TargetMeanEncoder':
        
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
                X_jg_y = dataArr[X_jg, y_j].astype(float)
                
                levelMeans[j][g] = (
                    np.mean(X_jg_y) 
                    if not np.isnan(np.mean(X_jg_y))
                    else grandMean
                )
                levelCounts[j][g] = dataArr[X_jg].shape[0]
                
        if self.priorSize is not None or self.priorFrac is not None:
            
            levelMeansSmoothed = {j: {} for j in range(X.shape[1])}
            
            for j in levelMeans.keys():
                for g in levelMeans[j].keys():
                    
                    if self.priorSize is not None:
                        weightSmoothed = self.priorSize / (self.priorSize + levelCounts[j][g])
                    elif self.priorFrac is not None:
                        priorSize = X.shape[0]*self.priorFrac
                        weightSmoothed = priorSize / (priorSize + levelCounts[j][g])
                        
                    levelMeansSmoothed_jg = (
                        (1 - weightSmoothed)*levelMeans[j][g] + weightSmoothed*grandMean
                    )                        
                    levelMeansSmoothed[j][g] = (
                        levelMeansSmoothed_jg
                        if not np.isnan(levelMeansSmoothed_jg)
                        else grandMean
                    )     
            self.levelMeansSmoothed = levelMeansSmoothed
            
        self.levelMeans = levelMeans
        self.grandMean = grandMean                
        return self
        
    def transform(self, X: np.array) -> np.array:
        
        X = X.values if isinstance(X, pd.DataFrame) else X
        XTransformed = np.empty_like(X).astype(float)
        
        for j in range(XTransformed.shape[1]):
            
            if self.priorSize is not None or self.priorFrac is not None:
                meansGetter = np.vectorize(self.levelMeansSmoothed[j].get)
            else:
                meansGetter = np.vectorize(self.levelMeans[j].get)
                
            XTransformed[:, j] = meansGetter(X[:, j])
            
            # imputes for levels in validation set not seen in training set
            XTransformed[np.isnan(XTransformed[:, j]), j] = self.grandMean
                
        return XTransformed