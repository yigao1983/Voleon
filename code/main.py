import glob
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import HeteroLinearModel as HLM

def read_data(csv_file):
    
    try:
        df = pd.read_csv(csv_file)
        X = df.x.values.reshape(len(df), 1)
        y = df.y.values
        
        return X, y
        
    except Exception as e:
        print(e)

def peek_pattern(csv_file_lst):
    
    markers = ("o", "s", "x", "d", "^")
    colors  = ("k", "b", "g", "r", "c")
    
    ols_df = pd.DataFrame(columns=["File name", "a", "b"])
    
    plt.figure()
    X_all = []
    err_all = []
    for idx, csv_file in enumerate(csv_file_lst):
        X, y = read_data(csv_file)
        ols = lm.LinearRegression().fit(X, y)
        err = y - ols.predict(X)
        
        ols_df.loc[idx] = pd.Series([csv_file.split("/")[-1], ols.coef_.item(), ols.intercept_], \
                                    index=ols_df.columns)
        
        plt.scatter(X, err/np.abs(X.ravel())**0.5, \
                    marker=markers[idx], color=colors[idx], label="data_1_{}".format(idx+1))
        
        X_all.extend(X.ravel())
        err_all.extend(err)
    
    ols_df.to_csv("ols.csv")
    
    plt.xlabel(r"$x$")
    plt.ylabel(r"OLS residual / $|x|^{1/2}$")
    plt.legend(loc=2, frameon=False)
    plt.savefig("peek_pattern.pdf", bbox_inches="tight")
    
    X_all   = np.array(X_all)
    err_all = np.array(err_all)
    XX_all  = np.vstack((np.abs(X_all), X_all**2))
    
    model = sm.OLS(err_all**2, sm.add_constant(XX_all.T))
    result = model.fit()
    
    print(result.summary())
    
def evaluate_params(csv_files):
    
    hlm_df = pd.DataFrame(columns=["File name", "a", "b"])
    
    for idx, csv_file in enumerate(csv_file_lst):
        X, y = read_data(csv_file)
        hlm = HLM.HeteroLinearModel().fit(X, y)
        hlm_df.loc[idx] = pd.Series([csv_file.split("/")[-1], hlm.coef_, hlm.intercept_], \
                                    index=hlm_df.columns)
        print(hlm.multiplicity_)
    
    hlm_df.to_csv("hlm.csv")
    
if __name__ == "__main__":
    
    csv_file_lst = sorted([csv for csv in glob.glob("../*.csv")])
    peek_pattern(csv_file_lst)
    evaluate_params(csv_file_lst)
