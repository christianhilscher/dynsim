import numpy as np
import pandas as pd
import pathlib
import os

import matplotlib.pyplot as plt

from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.transform import factor_cmap, dodge
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
palette = ["#c9d9d3", "#718dbf", "#e84d60", "#648450"]

def make_df(dataf, var):
    dataf = dataf.copy()

    j = 0
    ahead_ls = np.arange(1, len(dataf["period_ahead"].unique()), 4)
    out = pd.DataFrame(columns=["ahead", "frac_ml", "frac_standard"])
    for ahead in ahead_ls:
        df_ana = dataf[dataf["period_ahead"]==ahead]

        df_ana["frac_ml"] = df_ana[var+"_real"] == df_ana[var+"_ml"]
        df_ana["frac_standard"] = df_ana[var+"_real"] == df_ana[var+"_standard"]
        df_ana["frac_ml"] = df_ana["frac_ml"].astype(int)
        df_ana["frac_standard"] = df_ana["frac_standard"].astype(int)

        out.loc[j, "ahead"] = ahead
        out.loc[j, "frac_ml"] = df_ana["frac_ml"].mean()
        out.loc[j, "frac_standard"] = df_ana["frac_standard"].mean()
        j += 1

    out.set_index("ahead", inplace=True)
    out = pd.DataFrame(out.unstack())
    out.reset_index(inplace=True)

    out.rename(columns={"level_0": "type", "ahead": "years", 0: "value"}, inplace=True)
    return out



def make_plot(dataf, var):
    dataf = dataf.copy()
    abc = make_df(df, var)

    years = abc["years"].unique().tolist()
    years = [str(year) for year in years]
    types = abc["type"].unique().tolist()
    value = abc["value"].tolist()

    x = [ (year, type) for year in years for type in types]
    counts = value

    source = ColumnDataSource(data=dict(x=x, counts=counts))


    name = "Fraction of correct predictions for " + str(var)
    p = figure(x_range=FactorRange(*x), plot_height=450, title=name)

    p.vbar(x='x', top='counts', width=0.9, source=source, fill_color=factor_cmap('x', palette=palette, factors=types, start=1, end=2))

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    return p

var = "working"

pic = make_plot(df, var)
output_file(output_week + var + ".html")
show(pic)
