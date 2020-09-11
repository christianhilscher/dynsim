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

def get_data(dataf, into_future, variable, metric):
    dataf = dataf.copy()
    dataf = dataf[dataf[variable + "_real"] > 0]

    ml_values = np.empty(len(into_future))
    standard_values = np.empty_like(ml_values)
    real_values = np.empty_like(ml_values)

    for (ind, a) in enumerate(into_future):
        df_ana = dataf[dataf["period_ahead"] == a]

        ml_values[ind] = get_value(df_ana, variable, metric, type="_ml")
        standard_values[ind] = get_value(df_ana, variable, metric, type="_standard")
        real_values[ind] = get_value(df_ana, variable, metric, type="_real")

    dici = {"ml_values": ml_values,
            "standard_values": standard_values,
            "real_values": real_values}
    dici["ahead"] = into_future
    return dici

def get_value(dataf, var, metric, type):
    dataf = dataf.copy()
    variable = var + type

    if metric == "mean":
        res = dataf[variable].mean()
    elif metric == "median":
        res = dataf[variable].median()
    elif metric == "variance":
        res = dataf[variable].var()
    elif metric == "p90p50":
        p90_val = np.quantile(dataf[variable], 0.9)
        p50_val = np.quantile(dataf[variable], 0.5)
        res = p90_val/p50_val
    elif metric == "p50p10":
        p10_val = np.quantile(dataf[variable], 0.1)
        p50_val = np.quantile(dataf[variable], 0.5)
        res = p50_val/p10_val
    else:
        pass
    return res

def plot_deviations(dataf, into_future, variable, metric):
    dataf = dataf.copy()

    dataf = pd.DataFrame(dataf)
    dataf = pd.melt(dataf, id_vars=["ahead"])

    future = dataf["ahead"].unique().tolist()
    future = [str(f) for f in future]
    types = dataf["variable"].unique().tolist()
    x = [(a, type) for a in future for type in types]

    counts = dataf["value"]

    name = metric + " for " + variable

    s = ColumnDataSource(data=dict(x=x, counts=counts))
    p = figure(x_range=FactorRange(*x), title=name)
    p.vbar(x='x', top='counts', width=0.9, source=s,fill_color=factor_cmap('x', palette=Spectral6, factors=types, start=1, end=2))
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    return p


df = pd.read_pickle(output_week + "df_analysis")

ahead = np.arange(1, len(df["period_ahead"].unique()), 5)

variable = "gross_earnings"

metrics = ["mean", "median", "variance", "p90p50", "p50p10"]

plist = []
res_dici = {}
for m in metrics:
    abc = get_data(df, ahead, variable, m)
    plot = plot_deviations(abc, ahead, variable, m)
    res_dici[m] = abc
    plist.append(plot)

grid = gridplot([[plist[0], plist[1], plist[2], plist[3], plist[4]]], plot_width=400, plot_height=800)
output_file(output_week + variable + ".html")
show(grid)
