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
sim_path = "/Users/christianhilscher/desktop/dynsim/src/sim/"

current_week = "43"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)

def transition_probability(df, type, value_t, value_t1):

    return (sum((df["employment_status_" + type]==value_t) &
                (df["employment_status_t1_" + type]==value_t1))/len(df))



def marginal_probability(df, type, value_t, t=True):

    if t==True:
        return (sum((df["employment_status_" + type]==value_t))/len(df))
    else:
        return (sum((df["employment_status_t1_" + type]==value_t))/len(df))

def make_matrix(df, type, title):

    n_stati = 4
    trans_mat = np.zeros((n_stati + 1, n_stati + 1))

    for i in np.arange(n_stati):
        for j in np.arange(n_stati):
            trans_mat[i, j] = transition_probability(df, type, i, j)

    for i in np.arange(n_stati):
            trans_mat[i, -1] = marginal_probability(df, type, i, t=True)

    for j in np.arange(n_stati):
            trans_mat[-1, j] = marginal_probability(df, type, j, t=False)

    assert(
        np.round(sum(trans_mat[-1,]), decimals=2) == np.round(sum(trans_mat[:,-1]), decimals=2) == 1
    )
    trans_mat[-1, -1] = sum(trans_mat[-1,])

    print(title + ": " + str(len(df)))
    pd.DataFrame(trans_mat).to_csv(title + "_" + type + ".csv", index=False)

dataf = pd.read_pickle(input_path + "merged")
dataf1 = pd.read_pickle(output_week + "df_analysis_full")

os.chdir(output_week + "trans_mat/")


ts = ["real", "ml", "standard"]
for t in ts:
    make_matrix(dataf1, t, "cohort")
    make_matrix(dataf1[dataf1["female_real"]==1], t, "cohort_female")
    make_matrix(dataf1[dataf1["female_real"]==0], t, "cohort_male")

    make_matrix(dataf1[[30 <= age < 35 for age in dataf1["age_real"]]&(dataf1["female_real"]==1)], t, "cohort_3034_female")
    make_matrix(dataf1[[35 <= age < 40 for age in dataf1["age_real"]]&(dataf1["female_real"]==1)], t, "cohort_3539_female")
    make_matrix(dataf1[[40 <= age < 45 for age in dataf1["age_real"]]&(dataf1["female_real"]==1)], t, "cohort_4044_female")
    make_matrix(dataf1[[45 <= age < 50 for age in dataf1["age_real"]]&(dataf1["female_real"]==1)], t, "cohort_4549_female")
    make_matrix(dataf1[[50 <= age < 55 for age in dataf1["age_real"]]&(dataf1["female_real"]==1)], t, "cohort_5054_female")
    make_matrix(dataf1[[55 <= age < 60 for age in dataf1["age_real"]]&(dataf1["female_real"]==1)], t, "cohort_5559_female")
    make_matrix(dataf1[[60 <= age < 65 for age in dataf1["age_real"]]&(dataf1["female_real"]==1)], t, "cohort_6064_female")
    make_matrix(dataf1[[65 <= age < 70 for age in dataf1["age_real"]]&(dataf1["female_real"]==1)], t, "cohort_6570_female")

    make_matrix(dataf1[[30 <= age < 35 for age in dataf1["age_real"]]&(dataf1["female_real"]==0)], t, "cohort_3034_male")
    make_matrix(dataf1[[35 <= age < 40 for age in dataf1["age_real"]]&(dataf1["female_real"]==0)], t, "cohort_3539_male")
    make_matrix(dataf1[[40 <= age < 45 for age in dataf1["age_real"]]&(dataf1["female_real"]==0)], t, "cohort_4044_male")
    make_matrix(dataf1[[45 <= age < 50 for age in dataf1["age_real"]]&(dataf1["female_real"]==0)], t, "cohort_4549_male")
    make_matrix(dataf1[[50 <= age < 55 for age in dataf1["age_real"]]&(dataf1["female_real"]==0)], t, "cohort_5054_male")
    make_matrix(dataf1[[55 <= age < 60 for age in dataf1["age_real"]]&(dataf1["female_real"]==0)], t, "cohort_5559_male")
    make_matrix(dataf1[[60 <= age < 65 for age in dataf1["age_real"]]&(dataf1["female_real"]==0)], t, "cohort_6064_male")
    make_matrix(dataf1[[65 <= age < 70 for age in dataf1["age_real"]]&(dataf1["female_real"]==0)], t, "cohort_6570_male")
