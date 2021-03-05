import numpy as np
import pandas as pd
import pathlib
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from bokeh.plotting import show, output_file
###############################################################################
current_week = "38"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
plot_path = "/Users/christianhilscher/Desktop/dynsim/src/plotting/"
os.chdir(plot_path)
###############################################################################
from aux_plots import sizeplot, countplot, variable_means, plot_age_employment, plot_cohort_employment
dataf = pd.read_pickle(output_path + "filled_dici_illmitz_est_reduced.pkl")

###############################################################################
# Comparing variable means
var_list = ['working', 'fulltime', 'hours', 'gross_earnings', 'birth', 'married']

variable_means(dataf, var_list, output_week)

# Size and retention rate
fig_count = countplot(dataf)
fig_count.savefig(output_week + "duration.png")

fig_size = sizeplot(dataf)
fig_size.savefig(output_week + "size.png")
###############################################################################
df = dataf['ml']
age_plot = plot_age_employment(df)
output_file(output_week + "age_employment.html", title="Employment by age")
show(age_plot)

cohort_plot = plot_cohort_employment(df)
output_file(output_week + "cohort_employment.html", title="Employment by cohort")
show(cohort_plot)
