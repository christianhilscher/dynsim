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
current_week = "36"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
plot_path = "/Users/christianhilscher/Desktop/dynsim/src/plotting/"
os.chdir(plot_path)

df = pd.read_pickle(output_week + "df_analysis")


title1 = "How many years we observe people in our sample"
abc = df.groupby("pid")["period_ahead"].max()
hist, edges = np.histogram(abc, 32)
hist_df = pd.DataFrame({"obs": hist,
                        "left": edges[:-1],
                        "right": edges[1:]})
hist_df["bottom"] = 0
src = ColumnDataSource(hist_df)
p = figure(plot_height = 600, plot_width = 600,
              y_axis_label = "Count",
              title=title1)

p.quad(bottom = "bottom", top = "obs",left = "left", right = "right", source = src, fill_alpha=0.5, fill_color=Spectral6[0], legend_label="people")
show(p)



title2 = "People added to our sample in year..."
abc = df.groupby("pid")["year"].min()
hist, edges = np.histogram(abc, 32)
hist_df = pd.DataFrame({"obs": hist,
                        "left": edges[:-1],
                        "right": edges[1:]})
hist_df["bottom"] = 0
src = ColumnDataSource(hist_df)
p = figure(plot_height = 600, plot_width = 600,
              y_axis_label = "Count",
              title=title2)

p.quad(bottom = "bottom", top = "obs",left = "left", right = "right", source = src, fill_alpha=0.5, fill_color=Spectral6[0], legend_label="people")
show(p)



title3 = "Education of our sample"
abc = df.groupby("pid")["education_real"].max()
hist, edges = np.histogram(abc, 32)
hist_df = pd.DataFrame({"obs": hist,
                        "left": edges[:-1],
                        "right": edges[1:]})
hist_df["bottom"] = 0
src = ColumnDataSource(hist_df)
p = figure(plot_height = 600, plot_width = 600,
              y_axis_label = "Count",
              title=title3)

p.quad(bottom = "bottom", top = "obs",left = "left", right = "right", source = src, fill_alpha=0.5, fill_color=Spectral6[0], legend_label="people")
show(p)


title4 = "Gender: 0 = male; 1 = female"
abc = df.groupby("pid")["female_real"].mean()
hist, edges = np.histogram(abc, 32)
hist_df = pd.DataFrame({"obs": hist,
                        "left": edges[:-1],
                        "right": edges[1:]})
hist_df["bottom"] = 0
src = ColumnDataSource(hist_df)
p = figure(plot_height = 600, plot_width = 600,
              y_axis_label = "Count",
              title=title4)

p.quad(bottom = "bottom", top = "obs",left = "left", right = "right", source = src, fill_alpha=0.5, fill_color=Spectral6[0], legend_label="people")
show(p)


title5 = "Number of children"
abc = df.groupby("hid_real")["n_children_real"].max()
hist, edges = np.histogram(abc, 32)
hist_df = pd.DataFrame({"obs": hist,
                        "left": edges[:-1],
                        "right": edges[1:]})
hist_df["bottom"] = 0
src = ColumnDataSource(hist_df)
p = figure(plot_height = 600, plot_width = 600,
              y_axis_label = "Count",
              title=title5)

p.quad(bottom = "bottom", top = "obs",left = "left", right = "right", source = src, fill_alpha=0.5, fill_color=Spectral6[0], legend_label="n_children")
show(p)
