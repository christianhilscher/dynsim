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
current_week = 38
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
plot_path = "/Users/christianhilscher/Desktop/dynsim/src/plotting/"
os.chdir(plot_path)

palette = ["#c9d9d3", "#718dbf", "#e84d60", "#648450"]

def get_data(dataf, into_future, variable, metric):
    dataf = dataf.copy()

    dataf = dataf[dataf["gross_earnings_real"]>0]

    diff_ml = []
    diff_standard = []
    diff_real = []
    for ahead in into_future:
        df_ana = df_ana = dataf[dataf["period_ahead"]==ahead]
        diff_ml.append(_get_devs(df_ana, ahead, variable)[0])
        diff_standard.append(_get_devs(df_ana, ahead, variable)[1])
        diff_real.append(_get_devs(df_ana, ahead, variable)[2])

    devs_ml = []
    devs_standard = []
    devs_real = []
    if metric == "mean":
        for i, val in enumerate(diff_ml):
            devs_ml.append(np.abs(diff_ml[i].mean()))
            devs_standard.append(np.abs(diff_standard[i].mean()))
            devs_real.append(np.abs(diff_real[i].mean()))

        dici = {"ml_value": devs_ml,
                "standard_value": devs_standard,
                "real_value": devs_real}
    elif metric == "variance":
        for i, val in enumerate(diff_ml):
            devs_ml.append(diff_ml[i].var())
            devs_standard.append(diff_standard[i].var())
            devs_real.append(diff_real[i].var())

        dici = {"ml_value": devs_ml,
                "standard_value": devs_standard,
                "real_value": devs_real}
    elif metric == "median":
        for i, val in enumerate(diff_ml):
            devs_ml.append(np.abs(diff_ml[i].median()))
            devs_standard.append(np.abs(diff_standard[i].median()))
            devs_real.append(np.abs(diff_real[i].median()))

        dici = {"ml_value": devs_ml,
                "standard_value": devs_standard,
                "real_value": devs_real}
    else:
            for i, val in enumerate(diff_ml):
                devs_ml.append(_get_percentiles(diff_ml[i], diff_standard[i], diff_real[i], metric)[0])
                devs_standard.append(_get_percentiles(diff_ml[i], diff_standard[i], diff_real[i], metric)[1])
                devs_real.append(_get_percentiles(diff_ml[i], diff_standard[i], diff_real[i], metric)[2])

            dici = {"ml_value": devs_ml,
                    "standard_value": devs_standard,
                    "real_value": devs_real}
    dici["ahead"] = into_future

    return dici

def _get_devs(dataf, ahead, variable):
    dataf = dataf.copy()

    diff_ml = dataf[variable + "_ml"]
    diff_standard = dataf[variable + "_standard"]
    diff_real = dataf[variable + "_real"]

    return diff_ml, diff_standard, diff_real

def _get_percentiles(diff_ml, diff_standard, diff_real, metric):

    p90_ml = np.quantile(diff_ml, 0.9)
    p90_standard = np.quantile(diff_standard, 0.9)
    p90_real = np.quantile(diff_real, 0.9)

    p50_ml = np.quantile(diff_ml, 0.5)
    p50_standard = np.quantile(diff_standard, 0.5)
    p50_real = np.quantile(diff_real, 0.5)

    p10_ml = np.quantile(diff_ml, 0.1)
    p10_standard = np.quantile(diff_standard, 0.1)
    p10_real = np.quantile(diff_real, 0.1)

    if metric == "p90p50":
        ml_value = p90_ml / p50_ml
        standard_value = p90_standard / p50_standard
        real_value = p90_real / p50_real
    elif metric == "p50p10":
        ml_value = p50_ml / p10_ml
        standard_value = p50_standard / p10_standard
        real_value = p50_real / p10_real

    return np.abs(ml_value), np.abs(standard_value), np.abs(real_value)

def plot_deviations(dataf, into_future, variable, metric):
    dataf = dataf.copy()

    dataf = pd.DataFrame(dataf)
    dataf = pd.melt(dataf, id_vars=["ahead"])

    future = dataf["ahead"].unique().tolist()
    future = [str(f) for f in future]
    types = dataf["variable"].unique().tolist()
    x = [(a, type) for a in future for type in types]

    counts = dataf["value"]

    name = metric + " for " + variable + " | gross_earnings > 0"

    s = ColumnDataSource(data=dict(x=x, counts=counts))
    p = figure(x_range=FactorRange(*x), title=name)
    p.vbar(x='x', top='counts', width=0.9, source=s,fill_color=factor_cmap('x', palette=palette, factors=types, start=1, end=2))
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    return p

df = pd.read_pickle(output_week + "df_analysis_full")
df["year"].unique()
ahead = np.arange(1, len(df["period_ahead"].unique()), 4)

variable = "gross_earnings"

metrics = ["mean", "median", "variance"]

plist = []
for m in metrics:
    abc = get_data(df, ahead, variable, m)
    plot = plot_deviations(abc, ahead, variable, m)
    plist.append(plot)

grid = gridplot([[plist[0], plist[1], plist[2]]], plot_width=400, plot_height=600)
output_file(output_week + variable + ".html")
show(grid)

me = "p50p10"
abc = get_data(df, ahead, variable, me)
plot = plot_deviations(abc, ahead, variable, me)
show(plot)
