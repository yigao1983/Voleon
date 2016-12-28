import numpy as np
import sklearn.metrics as metrics

class HeteroLinearModel(object):
    
    def __init__(self, eta=1e-9):
        
        self.__eta = eta
        self.__a = 0
        self.__b = 0
        self.__c = 0
        
    @property
    def intercept_(self):
        return self.__b
        
    @property
    def coef_(self):
        return self.__a
        
    @property
    def multiplicity_(self):
        return self.__c
        
    def fit(self, X, y):
        
        a_11 = sum( X.ravel() / (np.abs(X.ravel()) + self.__eta) )
        a_12 = sum( 1. / (np.abs(X.ravel()) + self.__eta) )
        a_21 = sum( X.ravel()**2 / (np.abs(X.ravel()) + self.__eta) )
        a_22 = a_11
        
        z_1 = sum( y / (np.abs(X.ravel()) + self.__eta) )
        z_2 = sum( y * X.ravel() / (np.abs(X.ravel()) + self.__eta) )
        
        A = np.array([[a_11, a_12], [a_21, a_22]])
        z = np.array([z_1, z_2])
        
        self.__a, self.__b = np.linalg.solve(A, z)
        
        y_pred = self.predict(X)
        
        c2 = np.mean( (y - y_pred)**2 / (np.abs(X) + self.__eta) )
        
        self.__c = np.sqrt(c2)
        
        return self
        
    def predict(self, X):
        
        return self.__a * X.ravel() + self.__b
        
    def score(self, X, y):
        
        return metrics.r2_score(y, self.predict(X))
