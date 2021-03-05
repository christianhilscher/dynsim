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
current_week = "38"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
plot_path = "/Users/christianhilscher/Desktop/dynsim/src/plotting/"
os.chdir(plot_path)

palette = ["#c9d9d3", "#718dbf", "#e84d60", "#648450"]


def make_plot(dataf, into_future, variable):
    dataf = dataf.copy()
    # dataf = dataf[dataf[variable+"_real"]>0]

    diff_ml = np.empty(len(into_future))
    diff_standard = np.empty_like(diff_ml)
    real = np.empty_like(diff_ml)

    for i, ahead in enumerate(into_future):
        df_ana = dataf[dataf["period_ahead"]==ahead]
        real_value = df_ana[variable + "_real"].mean()
        ml_value = df_ana[variable + "_ml"].mean()
        standard_value = df_ana[variable+"_standard"].mean()

        diff_ml[i] = ml_value
        diff_standard[i] = standard_value
        real[i] = real_value

    future = [str(a) for a in into_future]
    types = ["ml", "standard", "real"]
    x = [(a, type) for a in future for type in types]

    counts = np.empty(len(diff_ml)*3)
    counts[::3] = diff_ml
    counts[1::3] = diff_standard
    counts[2::3] = real

    name = "Mean of " + variable

    s = ColumnDataSource(data=dict(x=x, counts=counts))
    p = figure(x_range=FactorRange(*x), title = name)
    p.vbar(x='x', top='counts', width=0.9, source=s,fill_color=factor_cmap('x', palette=palette, factors=types, start=1, end=2))
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    return p


df = pd.read_pickle(output_week + "df_analysis_full")

into_future = np.arange(1, len(df["period_ahead"].unique()), 4)
variable = "fulltime"

a = make_plot(df, into_future, variable)
output_file(output_week + variable + ".html")
show(a)
