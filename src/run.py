from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from sim.simulate import fill_dataf, predict
from estimation.standard import getdf

dir = Path.cwd().resolve().parent
input_path = dir / "input"
output_path = dir / "output"

###############################################################################
np.random.seed(2020)

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
