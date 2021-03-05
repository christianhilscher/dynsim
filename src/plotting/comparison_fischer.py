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
dir = Path(__file__).resolve().parents[2]
current_week = "week" + str(sys.argv[1])

path = dir / "output" / current_week
path.mkdir(parents=True, exist_ok=True)

###############################################################################


def restrict(dataf, working=False, female=None, max_age=None):
    
    dataf = dataf.copy()
    
    # Condition to include all or only working people 
    if working:
        condition_work = dataf["working_" + "real"] == 1
    else:
        condition_work = np.ones(len(dataf))
    
    # Including either all people, or only male and female
    if female == 1:
        condition_female = dataf["female_" + "real"] == 1
    elif female == 0:
        condition_female = dataf["female_" + "real"] == 0
    else:
        condition_female = np.ones(len(dataf))
        
    if type(max_age) == int:
        condition_age = dataf["age_" + "real"] <= max_age
    else:
        condition_age = np.ones(len(dataf))
        
    # Output is then sum of both conditions
    final_condition = (condition_female).astype(int) \
                        + (condition_work).astype(int) \
                        + (condition_age).astype(int)
                        
    df_out = dataf[final_condition == 3]
    
    return df_out

# Returns the AR(1) coefficient when estimated with a constant
def get_coeff(y, x):
    y = y.values
    x = x.values
    
    x = sm.add_constant(x)
    
    res = sm.OLS(y, x).fit()
    
    return res.params[1]
   
 
def calc_autocorr(dataf, variable):
    
    dataf = dataf.copy()
    
    # First sort age and make space for the coefficients
    age = np.sort(dataf["age_real"].unique())
    coeffs = np.empty(shape= (len(age), 3))
    
    # Looping through every typpe
    for (ind_t, typpe) in enumerate(["real", "standard", "ext"]):    
        
        var = variable + "_" + typpe
        var_lag = variable + "_t1_" + typpe
        
        # Calculate the coefficient for every age seperately
        for (ind_a, a) in enumerate(age):
            X = dataf.loc[dataf["age_" + typpe] == a, var_lag]
            Y = dataf.loc[dataf["age_" + typpe] == a, var]
            
            coeffs[ind_a, ind_t] = get_coeff(Y, X)
    
    # Concat all of them into a dataframe
    df_out = pd.DataFrame(data={"age": age,
                                "real": coeffs[:,0],
                                "standard": coeffs[:,1],
                                "ext": coeffs[:, 2]})
    
    return df_out


def plot(dataf, long_title, short_title):
    
    dataf = dataf.copy()
    source = ColumnDataSource(dataf)
    
    p = figure(title = long_title,
               y_range = (0, 1.3))
    
    p.line(x="age", y="real", source=source,
           line_color="black", line_dash = "solid", line_width=2,
           legend_label = "Real")
    
    p.line(x="age", y="standard", source=source,
           line_color="black", line_dash = "dashed", line_width=2,
           legend_label = "Standard")
    
    p.line(x="age", y="ext", source=source,
           line_color="black", line_dash = "dotted", line_width=2,
           legend_label = "Ext")
    
    # Adding axis labels
    p.xaxis.axis_label="Age"
    p.yaxis.axis_label="AR(1) coefficient"
    
    p = make_pretty(p)
    
    export_png(p, filename=str(path / short_title))
    

    
def make_pretty(p):
    p.xgrid.grid_line_color = None
    p.yaxis.minor_tick_line_width=0
    p.xaxis.minor_tick_line_width=0
    
    p.legend.location = "bottom_right"

    return p  
    
def get_names(variable, female=None):
    
    if female == 1:
        title = variable + " females"
        name = variable + "_female"
    elif female == 0:
        title = variable + " males"
        name = variable + "_male"
    else:
        title = variable + " all"
        name = variable + "_all"
        
    name = name + ".png"
    return title, name
        
##############################################################################
def plot_wrapper(dataf, variable, working=False, female=None, max_age=None):
    
    dataf = restrict(dataf, working, female, max_age)
    autocorr = calc_autocorr(dataf, variable)
    
    title, filename = get_names(variable, female)
    plot(autocorr, title, filename)
##############################################################################

if __name__ == "__main__":
    df = pd.read_pickle(path / "df_analysis_full")

    # Earnings
    plot_wrapper(df, "gross_earnings", working=True, max_age=65)
    plot_wrapper(df, "gross_earnings", working=True, max_age=65, female=0)
    plot_wrapper(df, "gross_earnings", working=True, max_age=65, female=1)
    
    # Hours
    plot_wrapper(df, "hours", working=True, max_age=65)
    plot_wrapper(df, "hours", working=True, max_age=65, female=0)
    plot_wrapper(df, "hours", working=True, max_age=65, female=1)
    
    
    # Fulltime
    plot_wrapper(df, "fulltime", working=True, max_age=65, female=0)
    plot_wrapper(df, "fulltime", working=True, max_age=65)
    plot_wrapper(df, "fulltime", working=True, max_age=65, female=1)
    
    # Working
    plot_wrapper(df, "working", working=False, max_age=65)
    plot_wrapper(df, "working", working=False, max_age=65, female=0)
    plot_wrapper(df, "working", working=False, max_age=65, female=1)