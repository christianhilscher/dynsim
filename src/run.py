from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from sim.simulate import fill_dataf, predict
from estimation.standard import getdf

dir = Path.cwd().parent
input_path = dir / "input"
output_path = dir / "output"
"""
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
"""
###############################################################################
np.random.seed(2021)

df = pd.read_pickle(input_path / 'merged').dropna()
df1 = getdf(df)

df1.sort_values(["pid", "year"], inplace=True)
df2 = df1.drop_duplicates(subset="pid", keep="first")


abc = fill_dataf(df1)
ghi = fill_dataf(df2)

pickle.dump(abc,
            open(output_path / "doc_full.pkl", "wb"))

pickle.dump(ghi,
            open(output_path / "doc_full2.pkl", "wb"))
