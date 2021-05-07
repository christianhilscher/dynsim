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

from aux_functions import cap_outliers
###############################################################################
dir = Path(__file__).resolve().parents[2]
current_week = "week" + str(sys.argv[1])

path = dir / "output" / current_week
path.mkdir(parents=True, exist_ok=True)
###############################################################################

def condition_by_type(dataf, method, working=False, female=None, max_age=None, ):
    
    dataf = dataf.copy()
    
    # Condition to include all or only working people 
    if working:
        condition_work = dataf["working_" + method] == 1
    else:
        condition_work = np.ones(len(dataf))
    
    # Including either all people, or only male and female
    if female == 1:
        condition_female = dataf["female_" + method] == 1
    elif female == 0:
        condition_female = dataf["female_" + method] == 0
    else:
        condition_female = np.ones(len(dataf))
    
    # Having an upper bound on age
    if type(max_age) == int:
        condition_age = dataf["age_" + method] <= max_age
    else:
        condition_age = np.ones(len(dataf))
        
    # Output is then sum of both conditions
    final_condition = (condition_female).astype(int) \
                        + (condition_work).astype(int) \
                        + (condition_age).astype(int)
                        
    df_out = dataf[final_condition == 3]
    
    return df_out

def restrict(dataf, working=False, female=None, max_age=None):
    
    dataf = dataf.copy()
    
    out_dici = {"real": condition_by_type(dataf, "real", working, female, max_age),
                "standard": condition_by_type(dataf, "standard", working, female, max_age),
                "ext": condition_by_type(dataf, "ext", working, female, max_age)}
    
    return out_dici

def calc_autocorr(dici, variable):

    # Space for coefficients
    
    # age = np.sort(dici["standard"]["age_standard"].unique())
    age = np.arange(dici["real"]["age_real"].min(),
                    dici["real"]["age_real"].max()+1)
    
    coeffs = np.zeros(shape=(len(age), 4))
    
    # Overall coeff:
    coef = get_coeff(dici["real"][variable + "_real"], 
                     dici["real"][variable + "_t1_real"])
    coeff_arr = np.repeat(coef, len(age))
    
    for (ind_t, method) in enumerate(["real", "standard", "ext"]):
        
        dataf_use = dici[method]
        var = variable + "_" + method
        var_lag = variable + "_t1_" + method
        
        # age_method = np.sort(dici[method]["age_" + method].unique())
        # Calculate the coefficient for every age seperately
        for (ind_a, a) in enumerate(age):
            X = dataf_use.loc[dataf_use["age_" + method] == a, var_lag]
            Y = dataf_use.loc[dataf_use["age_" + method] == a, var]
            
            coeffs[ind_a, ind_t] = get_coeff(Y, X)
    
    # Concat all of them into a dataframe
    df_out = pd.DataFrame(data={"age": age,
                                "real": coeffs[:,0],
                                "standard": coeffs[:,1],
                                "ext": coeffs[:, 2],
                                "coef": coeff_arr})
    
    return df_out

# Returns the AR(1) coefficient when estimated with a constant
def get_coeff(y, x):
    y = y.values
    x = x.values
    
    x = sm.add_constant(x)
    
    res = sm.OLS(y, x).fit()
    
    return res.params[-1]

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
    
    p.line(x="age", y="coef", source=source,
        line_color="red", line_dash = "solid", line_width=2,
        legend_label = "Real overall value")
    
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
    df = pd.read_pickle(path / "df_analysis_cohort")
    
    m_list = ["real", "standard", "ext"]
    df = cap_outliers(df, m_list)
    df = df[df["age_real"]<66]

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


