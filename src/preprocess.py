import numpy as np
import pandas as pd
import warnings
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

def read_file(path1, path2):
    
    df_test = pd.read_csv(path2)
    df_train = pd.read_csv(path1)
    id_ = df_test.id

    y_train = df_train.Tm
    df_train.drop("Tm",axis = 1, inplace = True)
    df_train.drop("SMILES", axis = 1, inplace = True )
    df_test.drop("SMILES", axis = 1, inplace = True )

    df_test.drop("id", axis = 1, inplace = True )
    df_train.drop("id", axis = 1, inplace = True )

    return df_test, df_train, id_, y_train



