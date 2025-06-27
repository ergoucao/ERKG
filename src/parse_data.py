import pandas as pd
import numpy as np
import torch

from profile_clustering import auto_cluster_k_shape

def parse_electricity(electricity_path:str, electrcity_type_path:str):
    all_elect = pd.read_csv(electricity_path, index_col=0)
    all_elect.index=pd.to_datetime(all_elect.index,unit='s')
    all_elect = all_elect.resample("1h", origin='start').sum()
    all_elect.to_csv("../data/AMPds2/Electricity_P_1h.csv")
    load_type=status_cluster(all_elect)
    print(load_type.shape[0])
    print("done !")

def status_cluster(all_elect:pd.DataFrame, cluster_method='k_means'):
    if cluster_method=='k_means':
        return auto_cluster_k_shape(all_elect)
    return 'error'

#     tmp=np.array(all_elect)
if __name__=='__main__':
    parse_electricity(electricity_path="../data/AMPds2/Electricity_P.csv",
                      electrcity_type_path="../data/AMPds2/load_type.csv")

