import numpy as np
import pandas as pd
from pathlib import Path

from bokeh.models import ColumnDataSource
from bokeh.io import export_png

from aux_functions import restrict

path = Path("/home/christian/dynasim/output/week10/")
df = pd.read_pickle("~/dynasim/output/week10/df_analysis_cohort")


abc = restrict(df, working=1)

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
            
    
    df_out = pd.DataFrame(tmp)
    return df_out

tmp = var_by_method_dici(abc, "gross_earnings", "year", "p90p50")
tmp

abc["real"]["gross_earnings_real"].quantile(0.9)

sum(df["birth_real"])
sum(df["birth_standard"])