import numpy as np
import pandas as pd
import pathlib
import os

import matplotlib.pyplot as plt

from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap
import matplotlib.pyplot as plt
###############################################################################
current_week = 36
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
plot_path = "/Users/christianhilscher/Desktop/dynsim/src/plotting/"
os.chdir(plot_path)

df = pd.read_pickle(output_week + "df_analysis")

df = df[(df["female_real"]==1)&(df["age_real"]<45)]


variable = "hours"
df["ml_deviation"] = np.abs(df[variable + "_real"] - df[variable + "_ml"])
df["standard_deviation"] = np.abs(df[variable + "_real"] - df[variable + "_standard"])

ahead = np.arange(1, len(df["period_ahead"].unique()), 5)

for a in ahead:
    df_ana = df[df["period_ahead"]==a]

    counts_ml = np.empty(5)
    counts_standard = np.empty_like(counts_ml)
    counts_real = np.empty_like(counts_ml)

    boundaries = [10, 20, 30, 40]


    counts_ml[0] = sum(df_ana["hours_ml"]<boundaries[0])
    counts_ml[1] = sum((df_ana["hours_ml"]>boundaries[0]) & (df_ana["hours_ml"]<boundaries[1]))
    counts_ml[2] = sum((df_ana["hours_ml"]>boundaries[1]) & (df_ana["hours_ml"]<boundaries[2]))
    counts_ml[3] = sum((df_ana["hours_ml"]>boundaries[2]) & (df_ana["hours_ml"]<boundaries[3]))
    counts_ml[4] = sum(df_ana["hours_ml"]>boundaries[3])

    counts_standard[0] = sum(df_ana["hours_standard"]<boundaries[0])
    counts_standard[1] = sum((df_ana["hours_standard"]>boundaries[0]) & (df_ana["hours_standard"]<boundaries[1]))
    counts_standard[2] = sum((df_ana["hours_standard"]>boundaries[1]) & (df_ana["hours_standard"]<boundaries[2]))
    counts_standard[3] = sum((df_ana["hours_standard"]>boundaries[2]) & (df_ana["hours_standard"]<boundaries[3]))
    counts_standard[4] = sum(df_ana["hours_standard"]>boundaries[3])

    counts_real[0] = sum(df_ana["hours_real"]<boundaries[0])
    counts_real[1] = sum((df_ana["hours_real"]>boundaries[0]) & (df_ana["hours_real"]<boundaries[1]))
    counts_real[2] = sum((df_ana["hours_real"]>boundaries[1]) & (df_ana["hours_real"]<boundaries[2]))
    counts_real[3] = sum((df_ana["hours_real"]>boundaries[2]) & (df_ana["hours_real"]<boundaries[3]))
    counts_real[4] = sum(df_ana["hours_real"]>boundaries[3])


    fig, axs = plt.subplots(5,5, figsize=(18,18), sharey="row")
    types = ["ml", "standard", "real"]


    for r in np.arange(0,5):
        axs[r, 0].bar(types, [counts_ml[r], counts_standard[r], counts_real[0]], color=["c", "b", "k"])

        axs[r, 1].bar(types, [counts_ml[r], counts_standard[r], counts_real[1]], color=["c", "b", "k"])

        axs[r, 2].bar(types, [counts_ml[r], counts_standard[r], counts_real[2]], color=["c", "b", "k"])

        axs[r, 3].bar(types, [counts_ml[r], counts_standard[r], counts_real[3]], color=["c", "b", "k"])

        axs[r, 4].bar(types, [counts_ml[r], counts_standard[r], counts_real[4]], color=["c", "b", "k"])

    axs[0,0].set_title("0-9 hours in reality")
    axs[0,1].set_title("9-19 hours in reality")
    axs[0,2].set_title("19-29 hours in reality")
    axs[0,3].set_title("29-39 hours in reality")
    axs[0,4].set_title("+40 hours in reality")

    axs[0,0].set_title("0-9 hours predicted")
    axs[1,0].set_title("9-19 hours predicted")
    axs[2,0].set_title("19-29 hours predicted")
    axs[3,0].set_title("29-39 hours predicted")
    axs[4,0].set_title("+40 hours predicted")
    name = "Crosstable for a " + str(a) + " year ahead prediction"
    fig.suptitle(name)
    plt.savefig(output_week + "/pngs/cross_" + str(a) + "_female_young.png")
 
