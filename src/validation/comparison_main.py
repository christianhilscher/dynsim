import sys
from pathlib import Path
import numpy as np
import pandas as pd

from aux_functions import plot_mean_by_age, plot_lifetime, wrap_employment_plots, wrap_income_age_plots, wrap_income_year_plots, wrap_inequality_year_plots

###############################################################################
dir = Path(__file__).resolve().parents[2]
current_week = "week" + str(sys.argv[1])

path = dir / "output" / current_week
path.mkdir(parents=True, exist_ok=True)


###############################################################################


###############################################################################
if __name__ == "__main__":
    
    df = pd.read_pickle(path / "df_analysis_cohort")
    
    m_list = ["real", "standard", "ext"]
    
    plot_mean_by_age(df, m_list, "married", path)
    plot_mean_by_age(df, m_list, "in_couple", path)
    plot_mean_by_age(df, m_list, "n_people", path)
    plot_mean_by_age(df, m_list, "n_children", path)
    plot_mean_by_age(df, m_list, "birth", path)
    
    # plot_lifetime(df, "real", path)
    # plot_lifetime(df, "standard", path)
    # plot_lifetime(df, "ext", path)
    
    # wrap_employment_plots(df, path)
    # wrap_income_age_plots(df, path)
    # wrap_income_year_plots(df, path)
    # wrap_inequality_year_plots(df, path)