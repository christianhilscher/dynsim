import numpy as np
import pandas as pd
import pickle
import os

from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show, gridplot
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Spectral6
from bokeh.transform import factor_cmap, dodge
##############################################################################
def quick_analysis(dataf):

    print("Data Types:")
    print(dataf.dtypes)
    print("Rows and Columns:")
    print(dataf.shape)
    print("Column Names:")
    print(dataf.columns)
    print("Null Values:")
    print(dataf.apply(lambda x: sum(x.isnull()) / len(dataf)))
##############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
sim_path = "/Users/christianhilscher/desktop/dynsim/src/sim/"

current_week = "47"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"

def make_cohort(dataf, birthyears):
    dataf = dataf.copy()

    birthyear = dataf["year"] - dataf["age_real"]
    condition = [by in birthyears for by in birthyear]
    dataf = dataf.loc[condition]
    dataf = dataf[dataf["east_real"]==0]


    return dataf

dataf = pd.read_pickle(input_path + "merged")
dataf1 = pd.read_pickle(output_week + "df_analysis_full")
# dataf2 = pd.read_pickle(output_week + "df_analysis_workingage")

palette = ["#c9d9d3", "#718dbf", "#e84d60", "#648450"]

cohorts = np.arange(1945, 1955)
df = make_cohort(dataf1, cohorts)

def plot_lifetime(df, type):
    print(type)
    ylist = []
    list0 = []
    list1 = []
    list2 = []
    list3 = []
    interv = np.sort(df["age_real"].unique())

    for a in interv:
        df_rel = df[df["age_real"]==a]
        n = len(df_rel)

        status0 = sum(df_rel["employment_status_" + type] == 0)/n
        status1 = sum(df_rel["employment_status_" + type] == 1)/n
        status2 = sum(df_rel["employment_status_" + type] == 2)/n
        status3 = sum(df_rel["employment_status_" + type] == 3)/n

        ylist.append(str(a))
        list0.append(status0)
        list1.append(status1)
        list2.append(status2)
        list3.append(status3)

    dici = {"age": ylist,
            "0": list0,
            "1": list1,
            "2": list2,
            "3": list3}

    #alllist = ["0", "1", "2", "3"]
    #labels = ["N.E.", "Rente", "Teilzeit", "Vollzeit"]
    alllist = ["3", "2", "0", "1"]
    labels = ["Vollzeit", "Teilzeit", "N.E.", "Rente"]


    p = figure(x_range=ylist, plot_height=250, plot_width=1500, title="Employment Status by age: West Germany / type: " + type)

    p.vbar_stack(alllist, x='age', width=0.9, color=palette, source=dici,
                 legend_label=labels)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "bottom_left"
    p.legend.orientation = "horizontal"

    show(p)

if __name__ == "__main__":
    plot_lifetime(dataf1, "real")
    plot_lifetime(dataf1, "ml")
    plot_lifetime(dataf1, "ext")
