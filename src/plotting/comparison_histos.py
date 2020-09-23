import numpy as np
import pandas as pd
import pathlib
import os

import matplotlib.pyplot as plt

from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
###############################################################################
current_week = "38"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
plot_path = "/Users/christianhilscher/Desktop/dynsim/src/plotting/"
os.chdir(plot_path)

df = pd.read_pickle(output_week + "df_analysis_full")

def remove_outliers(arr, low, high):
    arr = arr.copy()
    lower = np.quantile(arr, low)
    upper = np.quantile(arr, high)

    out = arr[np.logical_and(arr>lower, arr<upper)]
    return out


def _hist_df(arr):
    arr = arr.copy()
    arr = remove_outliers(arr, 0.05, 0.95)
    bins = 40
    hist, edges = np.histogram(arr, bins)
    hist_df = pd.DataFrame({"obs": hist,
                            "left": edges[:-1],
                            "right": edges[1:]})
    hist_df["bottom"] = 0
    return hist_df

def make_plot_earnings(dataf, ahead):
    dataf = dataf.copy()
    dataf = dataf[dataf["gross_earnings_real"]>0]
    df_ana = dataf[dataf["period_ahead"]==ahead]

    earnings_diff_ml = df_ana["gross_earnings_real"] - df_ana["gross_earnings_ml"]
    earnings_diff_standard = df_ana["gross_earnings_real"] - df_ana["gross_earnings_standard"]


    abc = _hist_df(earnings_diff_ml)
    ghi = _hist_df(earnings_diff_standard)

    abc = abc.add_suffix("_ml")
    ghi = ghi.add_suffix("_standard")

    hist_df = pd.concat([abc, ghi], axis=1)

    src = ColumnDataSource(hist_df)

    sze = len(df_ana)
    title = "gross_earnings: Errors with " + str(ahead) + " years ahead prediction. Sample Size: " + str(sze)
    p = figure(plot_height = 600, plot_width = 600,
                  y_axis_label = "Count", title=title)

    p.quad(bottom = "bottom_ml", top = "obs_ml",left = "left_ml", right = "right_ml", source = src, fill_alpha=0.5, fill_color=Spectral6[0], legend_label="ml")

    p.quad(bottom = "bottom_standard", top = "obs_standard",left = "left_standard", right = "right_standard", source = src, fill_alpha=0.5, fill_color=Spectral6[3], legend_label="standard")

    return p

def make_plts(dataf):
    dataf = dataf.copy()
    ahead_ls = np.arange(1, 25, 5)
    plts = []
    for a in ahead_ls:
        plts.append(make_plot_earnings(dataf, a))

    p = row(plts)
    return p


pic = make_plts(df)
output_file(output_week + "gross_earnings_difference" + ".html")
show(pic)
