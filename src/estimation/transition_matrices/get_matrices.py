import numpy as np
import pandas as pd
import pickle
import os, pathlib

import lightgbm as lgb
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap, dodge
##############################################################################
def quick_analysis(dataf):

    print("Data Types:")
    print(dataf.dtypes)
    print("Rows and Columns:")
    print(dataf.shape)
    print("Column Names:")
    print(dataf.columns)
    print("Null Values:")
    print(dataf.apply(lambda x: sum(x.isnull()) / len(dataf)))
##############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"

os.chdir(estimation_path)
from standard import getdf


def transition_probability(dataf, value_t, value_t1):

    return (sum((dataf["employment_status"]==value_t) &
                (dataf["employment_status_t1"]==value_t1))/len(dataf))



def marginal_probability(dataf, value_t, t=True):

    if t==True:
        return (sum((dataf["employment_status"]==value_t))/len(dataf))
    else:
        return (sum((dataf["employment_status_t1"]==value_t))/len(dataf))

def make_matrix(dataf):
    dataf = dataf.copy()

    n_stati = 4
    trans_mat = np.zeros((n_stati + 1, n_stati + 1))

    # Probabilities
    for i in np.arange(n_stati):
        for j in np.arange(n_stati):
            trans_mat[i, j] = transition_probability(dataf, i, j)

    # Marginal probabilities
    for i in np.arange(n_stati):
            trans_mat[i, -1] = marginal_probability(dataf, i, t=True)

    for j in np.arange(n_stati):
            trans_mat[-1, j] = marginal_probability(dataf, j, t=False)

    assert(
        np.round(sum(trans_mat[-1,]), decimals=2) == np.round(sum(trans_mat[:,-1]), decimals=2) == 1
    ), ValueError("Marginal probabilities don't aff up to 1.")
    trans_mat[-1, -1] = sum(trans_mat[-1,])


    return pd.DataFrame(trans_mat)

def write_matrices(dataf):
    dataf = dataf.copy()

    bins = np.arange(0, 101, 5)
    dataf["age_bin"] = pd.cut(dataf["age"], bins)

    dataf.sort_values("age", inplace=True)

    dici = {}

    for bin in dataf["age_bin"].unique():
        print("Age bracket starting with ", bin.left)

        for sex in ["female", "male"]:

            if sex == "female":
                df_tmp = dataf[(dataf["age_bin"] == bin)&(dataf["female"]==1)]
            else:
                df_tmp = dataf[(dataf["age_bin"] == bin)&(dataf["female"]==0)]

            title = sex + "_" + str(bin.left)
            dici[title] = make_matrix(df_tmp)
    return dici

df = pd.read_pickle(input_path + "merged")
df1 = getdf(df)
trans_matrices = write_matrices(df1)

pickle.dump(trans_matrices,
            open("transition_matrices/full_sample", "wb"))
