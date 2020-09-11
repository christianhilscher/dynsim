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
###############################################################################
current_week = 36
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
plot_path = "/Users/christianhilscher/Desktop/dynsim/src/plotting/"
os.chdir(plot_path)

def remove_outliers(arr, low, high):
    arr = arr.copy()
    lower = np.quantile(arr, low)
    upper = np.quantile(arr, high)

    out = arr[np.logical_and(arr>lower, arr<upper)]
    return out


def _hist_df(arr, bins):
    arr = arr.copy()
    arr = remove_outliers(arr, 0.05, 0.95)
    bins = bins
    hist, edges = np.histogram(arr, bins)
    hist_df = pd.DataFrame({"obs": hist,
                            "left": edges[:-1],
                            "right": edges[1:]})
    hist_df["bottom"] = 0
    return hist_df

def histo(dataf, var, type, binsize):
    dataf = dataf.copy()

    dataf = dataf[dataf[var]>0]

    if type == "ml":
        value = dataf[var + "_y"].to_numpy()
    elif type == "standard":
        value = dataf[var].to_numpy()
    elif type == "real":
        value = dataf[var + "_x"].to_numpy()

    histo_df = _hist_df(value, binsize)
    name = "Histogram of " + var + " with " + str(binsize) + " bins"
    s = ColumnDataSource(histo_df)
    p = figure(plot_height = 600, plot_width = 600,
                  y_axis_label = "Count", title=name)

    p.quad(bottom = "bottom", top = "obs",left = "left", right = "right", source = s, fill_alpha=0.5, legend_label=type)
    return p


df = pd.read_pickle(output_week + "df_analysis")

var = "hours"
types = ["ml", "standard", "real"]

plist = []
for type in types:
    plot = histo(df, var, type, 100)
    plist.append(plot)

grid = gridplot([[plist[0], plist[1], plist[2]]], plot_width=400, plot_height=600)
output_file(output_week + var + "_histogram.html")
show(grid)
