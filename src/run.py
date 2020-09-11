import numpy as np
import pandas as pd
import os
import pathlib
import pickle

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

cwd = os.getcwd()
sim_path = "/Users/christianhilscher/Desktop/dynsim/src/sim/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"

###############################################################################
os.chdir(sim_path)
from simulate import fill_dataf, predict

os.chdir(estimation_path)
from standard import getdf

os.chdir(cwd)

##############################################################################


df = pd.read_pickle(input_path + 'merged').dropna()
df1 = getdf(df)
df2 = df1.drop_duplicates(subset="pid")


abc = fill_dataf(df1)
ghi = fill_dataf(df2)

pickle.dump(abc,
            open(output_path + "doc_full.pkl", "wb"))

pickle.dump(ghi,
            open(output_path + "doc_full2.pkl", "wb"))
