import glob
import numpy as np
import matplotlib.pyplot as plt
import HeteroLinearModel as HLM

def peek_pattern(csv_file_lst, power=0.5):
    
    markers = ("o", "s", "x", "d", "^")
    colors  = ("k", "b", "g", "r", "c")
    
    plt.figure()
    for idx, csv_file in enumerate(csv_file_lst):
        hlm = HLM.HeteroLinearModel(csv_file).fit_ols()
        plt.scatter(hlm.X, (hlm.y-hlm.predict()) / np.power(np.abs(hlm.X.ravel()), power), \
                    color=colors[idx], marker=markers[idx], label=csv_file)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.show()    
    
if __name__ == "__main__":
    
    csv_file_lst = sorted([csv for csv in glob.glob("../*.csv")])
    
    peek_pattern(csv_file_lst, power=0.5)
