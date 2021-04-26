import sys
from pathlib import Path
import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource
from bokeh.io import export_png
from bokeh.plotting import figure

def plot_lifetime(df, type, path):

    df = df.copy()
    palette = ["#c9d9d3", "#718dbf", "#e84d60", "#648450"]
    
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
    
    str_path = "employment_" + type + ".png"

    export_png(p, filename=str(path/ str_path))

def var_by_method(dataf, variable):
    
    dataf_out = pd.DataFrame()
    dataf_out["pid"] = dataf["pid"]
    dataf_out["year"] = dataf["year"]
    dataf_out["hid"] = dataf["hid_real"]
    dataf_out["age"] = dataf["age_real"]

    for m in ["real", "standard", "ext"]:

        dataf_out[m] = dataf[variable + "_" + m]

    return dataf_out

def plot_mean_by_age(dataf, m_list, variable, path):
    
    dataf = dataf.copy()
    df = var_by_method(dataf, variable)
    df_plot = df.groupby("age")[m_list].mean()
    
    fig_title = variable
    file_title = variable + ".png"
    
    # return df_plot
    plot_age(df_plot, fig_title, file_title, path)


def make_pretty(p):
    p.xgrid.grid_line_color = None
    p.yaxis.minor_tick_line_width=0
    p.xaxis.minor_tick_line_width=0
    
    # p.legend.location = "bottom_right"

    return p

def plot_employment_status_by_age(dataf, employment_status, path, female=None, east=None):
    
    dataf = dataf.copy()
    
    dataf_rest = rest_dataf(dataf, female, east)
    
    status_list = ["N_E", "Rente", "Teilzeit", "Vollzeit"]
    status = status_list[employment_status]
    
    df_tmp = var_by_method(dataf_rest, "employment_status")
    tmp = df_tmp[["real", "standard", "ext"]] == employment_status
    df_plot = pd.concat([df_tmp["age"], tmp], axis=1)
    df_plot = df_plot.groupby("age").mean()
    
    # Plotting
    fig_title, file_title = get_titles(female, east, status)
    plot_age(df_plot, fig_title, file_title, path, interv=1)
    
def plot_age(dataf, fig_title, file_title, path, interv=0):
    
    source = ColumnDataSource(dataf)
    
    if interv==1:
        p = figure(title = fig_title, y_range=(0, 1))
    else:
        p = figure(title = fig_title)
    p.line(x="age", y="real", source=source,
        line_color="black", line_dash = "solid", line_width=2,
        legend_label = "Real")
        
    p.line(x="age", y="standard", source=source,
           line_color="black", line_dash = "dashed", line_width=2,
           legend_label = "Standard")
    
    p.line(x="age", y="ext", source=source,
           line_color="black", line_dash = "dotted", line_width=2,
           legend_label = "Ext")
    
    p.xaxis.axis_label = "Age"
    
    p = make_pretty(p)
    export_png(p, filename=str(path/ file_title))
    
def plot_year(dataf, fig_title, file_title, path, interv=0):
    
    source = ColumnDataSource(dataf)
    
    if interv==1:
        p = figure(title = fig_title, y_range=(0, 1))
    else:
        p = figure(title = fig_title)
    p.line(x="year", y="real", source=source,
        line_color="black", line_dash = "solid", line_width=2,
        legend_label = "Real")
        
    p.line(x="year", y="standard", source=source,
           line_color="black", line_dash = "dashed", line_width=2,
           legend_label = "Standard")
    
    p.line(x="year", y="ext", source=source,
           line_color="black", line_dash = "dotted", line_width=2,
           legend_label = "Ext")
    
    p.xaxis.axis_label = "Year"
    
    p = make_pretty(p)
    export_png(p, filename=str(path/ file_title))
    
    
def rest_dataf(dataf, female, east):
    
    dataf = dataf.copy()
    method = "real" # Gender and East do not change during the simulation
    
    # Including either all people, or only male and female
    if female == 1:
        condition_female = dataf["female_" + method] == 1
    elif female == 0:
        condition_female = dataf["female_" + method] == 0
    else:
        condition_female = np.ones(len(dataf))
        
    # Including either all people, or only east or west germans
    if east == 1:
        condition_east = dataf["east_" + method] == 1
    elif east == 0:
        condition_east = dataf["east_" + method] == 0
    else:
        condition_east = np.ones(len(dataf))
        
    # Output is then sum of both conditions
    final_condition = (condition_female).astype(int) \
                        + (condition_east).astype(int)
                        
    df_out = dataf[final_condition == 2]
    
    return df_out

def get_titles(female, east, status):
    
    title = ""
    shorttitle = status
    
    if (female==None) & (east==None):
        title = "Employment status: " + status + "; all people"
        shorttitle += "_mfew.png"
    elif (female==None) & (east==0):
        title = "Employment status: " + status + "; all genders, west Germany" 
        shorttitle += "_mfw.png"
    elif (female==None) & (east==1):
        title = "Employment status: " + status + "; all genders, east Germany"
        shorttitle += "_mfe.png"
    elif (female==0) & (east==None):
        title = "Employment status: " + status + "; male, whole Germany"
        shorttitle += "_mew.png"
    elif (female==1) & (east==None):
        title = "Employment status: " + status + "; female, whole Germany"
        shorttitle += "_few.png"
    elif (female==0) & (east==0):
        title = "Employment status: " + status + "; male, west Germany"
        shorttitle += "_mw.png"
    elif (female==0) & (east==1):
        title = "Employment status: " + status + "; male, east Germany"
        shorttitle += "_me.png"
    elif (female==1) & (east==0):
        title = "Employment status: " + status + "; female, west Germany"
        shorttitle += "_fw.png"
    elif (female==1) & (east==1):
        title = "Employment status: " + status + "; female, east Germany"
        shorttitle += "_fe.png"
    
    return title, shorttitle

def get_titles_incomes(suffix, variable, working, female, fulltime, measure):
    
    w_string = ""
    f_string = ""
    t_string = ""
    
    if working==1:
        w_string = "_working"
    else:
        pass
    
    if female==1:
        f_string = "_female"
    elif female==0:
        f_string = "_male"
    else:
        pass
        
    if fulltime==1:
        t_string = "_fulltime"
    elif fulltime==0:
        t_string = "_parttime"
    else:
        pass
    
    fig_title = suffix + measure + "_" + variable + w_string + f_string + t_string
    file_title = fig_title + ".png"
    
    return fig_title, file_title
    

def wrap_employment_plots(dataf, path):
    
    dataf = dataf.copy()
    
    for emp in np.arange(4):
        # All people, all employment status
        plot_employment_status_by_age(dataf, emp, path)
        
        # Males, all employment status
        plot_employment_status_by_age(dataf, emp, path, female=0)
        # Females, all employment status
        plot_employment_status_by_age(dataf, emp, path, female=1)
        
        # All_people, east Germany, all employment status
        plot_employment_status_by_age(dataf, emp, path, east=1)
        # All_people, west Germany, all employment status
        plot_employment_status_by_age(dataf, emp, path, east=0)
        
        # Males, east Germany, all employment status
        plot_employment_status_by_age(dataf, emp, path, female=0, east=1)
        # Males, west Germany, all employment status
        plot_employment_status_by_age(dataf, emp, path, female=0, east=0)
        # Females, east Germany, all employment status
        plot_employment_status_by_age(dataf, emp, path, female=1, east=1)
        # Females, west Germany, all employment status
        plot_employment_status_by_age(dataf, emp, path, female=1, east=0)
        
def condition_by_type(dataf, method, working=False, female=None, fulltime=None):
    
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
    
    # Including either all people, or only male and female
    if fulltime == 1:
        condition_fulltime = dataf["fulltime_" + method] == 1
    elif fulltime == 0:
        condition_fulltime = dataf["parttimetime_" + method] == 1
    else:
        condition_fulltime = np.ones(len(dataf))
        
    # Output is then sum of both conditions
    final_condition = (condition_female).astype(int) \
                        + (condition_work).astype(int) \
                        + (condition_fulltime).astype(int)
                        
    df_out = dataf[final_condition == 3]
    
    return df_out

def restrict(dataf, working=False, female=None, fulltime=None):
    
    dataf = dataf.copy()
    
    out_dici = {"real": condition_by_type(dataf, "real", working, female, fulltime),
                "standard": condition_by_type(dataf, "standard", working, female, fulltime),
                "ext": condition_by_type(dataf, "ext", working, female, fulltime)}
    
    return out_dici

def var_by_method_dici(dici, variable, group, measure):

    tmp = {}
    m_list = ["real", "standard", "ext"]
    for m in m_list:
        if measure == "mean":
            tmp[m] = dici[m].groupby(group)[variable + "_" + m].mean()
        elif measure == "median":
            tmp[m] = dici[m].groupby(group)[variable + "_" + m].median()
            
        elif measure == "p90p50":
            p90 = dici[m].groupby(group)[variable + "_" + m].quantile(0.9)
            p50 = dici[m].groupby(group)[variable + "_" + m].quantile(0.5)
            tmp[m] = p90/p50
        elif measure == "p90p10":
            p90 = dici[m].groupby(group)[variable + "_" + m].quantile(0.9)
            p10 = dici[m].groupby(group)[variable + "_" + m].quantile(0.1)
            tmp[m] = p90/p10
        elif measure == "p50p10":
            p50 = dici[m].groupby(group)[variable + "_" + m].quantile(0.5)
            p10 = dici[m].groupby(group)[variable + "_" + m].quantile(0.1)
            tmp[m] = p50/p10
        elif measure == "gini":
            tmp[m] = dici[m].groupby(group)[variable + "_" + m].agg(gini_coefficient)
            
    
    df_out = pd.DataFrame(tmp)
    return df_out

def plot_income_age(dataf, variable, path, working=None, female=None, fulltime=None, measure="mean"):

    dataf = dataf.copy()
    dici = restrict(dataf, working, female, fulltime)
    df_plot = var_by_method_dici(dici, variable, group="age_real", measure=measure)
    
    df_plot = df_plot.fillna(0)
    df_plot.reset_index(inplace=True)
    df_plot.rename(columns={"age_real": "age"}, inplace=True)
    
    fig_title, file_title = get_titles_incomes("age_", variable, working, female, fulltime, measure)
    plot_age(df_plot, fig_title, file_title, path)
    
def wrap_income_age_plots(dataf, path):
    
    dataf = dataf.copy()
    variables = ["gross_earnings", "hours"]
    for var in variables:
        for m in ["mean", "median"]:

            # All people
            plot_income_age(dataf, var, path=path, measure=m)
            plot_income_age(dataf, var, path=path, female=0, measure=m)
            plot_income_age(dataf, var, path=path, female=1, measure=m)
            
            # Conditional on working
            plot_income_age(dataf, var, path=path, working=1, measure=m)
            plot_income_age(dataf, var, path=path, working=1, female=0, measure=m)
            plot_income_age(dataf, var, path=path, working=1, female=1, measure=m)
            
            # Conditional on fulltime
            plot_income_age(dataf, var, path=path, fulltime=1, measure=m)
            plot_income_age(dataf, var, path=path, fulltime=1, female=0, measure=m)
            plot_income_age(dataf, var, path=path, fulltime=1, female=1, measure=m)
    
def plot_income_year(dataf, variable, path, working=None, female=None, fulltime=None, measure="mean"):
    
    dataf = dataf.copy()
    
    dici = restrict(dataf, working, female, fulltime)
    df_plot = var_by_method_dici(dici, variable, group="year", measure=measure)
    
    df_plot = df_plot.fillna(0)
    df_plot.reset_index(inplace=True)
    
    fig_title, file_title = get_titles_incomes("year_", variable, working, female, fulltime, measure)
    plot_year(df_plot, fig_title, file_title, path)
    
def wrap_income_year_plots(dataf, path):
    
    dataf = dataf.copy()
    variables = ["gross_earnings", "hours"]
    for var in variables:
        for m in ["mean", "median"]:
            
            # All people
            plot_income_year(dataf, var, path=path, measure=m)
            plot_income_year(dataf, var, path=path, female=0, measure=m)
            plot_income_year(dataf, var, path=path, female=1, measure=m)
            
            # Conditional on working
            plot_income_year(dataf, var, path=path, working=1, measure=m)
            plot_income_year(dataf, var, path=path, working=1, female=0, measure=m)
            plot_income_year(dataf, var, path=path, working=1, female=1, measure=m)
            
            # Conditional on fulltime
            plot_income_year(dataf, var, path=path, fulltime=1, measure=m)
            plot_income_year(dataf, var, path=path, fulltime=1, female=0, measure=m)
            plot_income_year(dataf, var, path=path, fulltime=1, female=1, measure=m)
        
def plot_inequality_year(dataf, variable, path, working=None, female=None, fulltime=None, measure="mean"):
    
    dataf = dataf.copy()
    
    dici = restrict(dataf, working, female, fulltime)
    df_plot = var_by_method_dici(dici, variable, group="year", measure=measure)
    
    df_plot = df_plot.fillna(0)
    df_plot.reset_index(inplace=True)
    
    fig_title, file_title = get_titles_incomes("ineq_", variable, working, female, fulltime, measure)
    plot_year(df_plot, fig_title, file_title, path)
    
def wrap_inequality_year_plots(dataf, path):
    
    dataf = dataf.copy()
    var = ["gross_earnings", "hours"]
    
    for v in var:
        for m in ["p90p50", "p90p10", "p50p10", "gini"]:
            
            plot_inequality_year(dataf, v, path, working=1, measure=m)
            plot_inequality_year(dataf, v, path, working=1, female=0,  measure=m)
            plot_inequality_year(dataf, v, path, working=1, female=1, measure=m)
   
def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))