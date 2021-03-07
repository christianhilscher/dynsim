import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys

import statsmodels.api as sm
from bokeh.layouts import row
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import export_png

###############################################################################
# dir = Path(__file__).resolve().parents[2]
# current_week = "week" + str(sys.argv[1])

# path = dir / "output" / current_week
# path.mkdir(parents=True, exist_ok=True)

df = pd.read_pickle("~/dynasim/output/week09/df_analysis_full")
###############################################################################

def var_by_method(dataf, variable):
    
    dataf_out = pd.DataFrame()
    dataf_out["pid"] = dataf["pid"]
    dataf_out["year"] = dataf["year"]
    dataf_out["hid"] = dataf["hid_real"]

    for m in ["real", "standard", "ext"]:

        dataf_out[m] = dataf[variable + "_" + m]

    return dataf_out

def var_by_age(dataf, variable):
    
    dataf_out = pd.DataFrame()
    dataf_out["age"] = np.sort(dataf["age_real"].unique())
    
    for m in ["real", "standard", "ext"]:
        dataf_out[m] = dataf.groupby("age_real")[variable + "_" + m].mean().values
    return dataf_out
        
def plot_by_age(dataf, variable):
    
    dataf_plot = var_by_age(dataf, variable)
    
    source = ColumnDataSource(dataf_plot)
    
    p = figure(title = variable)
    
    p.line(x="age", y="real", source=source,
           line_color="black", line_dash = "solid", line_width=2,
           legend_label = "Real")
    
    p.line(x="age", y="standard", source=source,
           line_color="black", line_dash = "dashed", line_width=2,
           legend_label = "Standard")
    
    p.line(x="age", y="ext", source=source,
           line_color="black", line_dash = "dotted", line_width=2,
           legend_label = "Ext")
    
    show(p)

df.columns.tolist()

##########

df_child = var_by_method(df, "child")
tmp = df_child.groupby("pid").max()
sum(tmp["real"]==tmp["standard"])/len(tmp)
sum(tmp["real"]==tmp["ext"])/len(tmp)
##########

len(df["hid_real"].unique())
len(df["hid_standard"].unique())
len(df["hid_ext"].unique())
##########

df_married = var_by_method(df, "married")
tmp = df_married.groupby("pid").max()
sum(tmp["real"]==tmp["standard"])/len(tmp)
sum(tmp["real"]==tmp["ext"])/len(tmp)
##########

df.groupby("age_real")["n_people_real", 
                       "n_people_standard",
                       "n_people_ext"].mean()

var_by_age(df, "n_children")

plot_by_age(df, "n_children")

np.unique(df["n_children_real"].values, return_counts=True)
np.unique(df["n_children_standard"].values, return_counts=True)
np.unique(df["n_children_ext"].values, return_counts=True)

df["n_children_real"].sum()
df["n_children_ext"].sum()


a = 3
a -= 1
a