import numpy as np
import pandas as pd
import pickle
import os, pathlib, importlib

from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap, dodge



def droplargest(dataf):
    dataf = dataf.copy()

    # Taking the largest difference in each period
    large_dev = dataf.groupby(["period_ahead"])["diff"].max().to_numpy()
    large_dev = large_dev[large_dev>0]

    # Condition that differnece == largest difference
    cond = [diff in large_dev for diff in dataf["diff"]]
    cond1 = [not t for t in cond]

    out = dataf[cond1]
    return out


def prepplot(dataf):
    dataf = dataf.copy()

    # Grouping by pid and period ahead
    df = dataf.groupby(["pid",
                        "period_ahead"])["gross_earnings_real",
                                         "gross_earnings_ext"].sum()


    df.reset_index(inplace=True)
    # Getting the absolute difference between predicted and real values
    df["diff"] = np.abs(df["gross_earnings_real"] - df["gross_earnings_ext"])

    df = droplargest(df)
    out = df.groupby("period_ahead")["diff"].describe()

    return out

def make_pretty(p):
    p.xgrid.grid_line_color = None
    p.yaxis.minor_tick_line_width=0
    p.xaxis.minor_tick_line_width=0

    return p


def plot_diff(dataf):
    source = ColumnDataSource(data=dataf)

    title = "Abolsute difference between estimated and real earnings"
    p = figure(title=title)

    # Adding lines
    p.line(x="period_ahead", y="25%", source=source,
           line_color="black", line_dash="solid", line_width=3,
           legend_label = "25 percentile")
    p.line(x="period_ahead", y="50%", source=source,
           line_color="black", line_dash="dashed", line_width=3,
           legend_label = "50 percentile")
    p.line(x="period_ahead", y="75%", source=source,
           line_color="black", line_dash="dotted", line_width=3,
           legend_label = "75 percentile")

    # Adding axis labels
    p.xaxis.axis_label="Periods predicted ahead"
    p.yaxis.axis_label="Difference real and estimated"

    # Minor adjustments for a marginally prettier graph
    p = make_pretty(p)
    show(p)


current_week = "47"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
dataf1 = pd.read_pickle(output_week + "df_analysis_full")


plot_data = prepplot(dataf1)
plot_diff(plot_data)
