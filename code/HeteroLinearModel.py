import numpy as np
import pandas as pd
import sklearn.linear_model as lm

class HeteroLinearModel(object):
    
    def __init__(self, csv_file):
        
        df = pd.read_csv(csv_file)
        
        self.__X = df.x.values.reshape(len(df), 1)
        self.__y = df.y.values
        
    @property
    def X(self):
        
        return self.__X
        
    @property
    def y(self):
        
        return self.__y
        
    def fit_ols(self):
        
        self.__ols = lm.LinearRegression(fit_intercept=True, normalize=True).fit(self.__X, self.__y)
        
        return self
        
    def predict(self):
        
        return self.__ols.predict(self.__X)
