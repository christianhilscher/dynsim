import numpy as np
import pandas as pd
import pathlib
import os


###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"
###############################################################################

# Loading data
dici_full = pd.read_pickle(output_path + "doc_full.pkl")
dici_est = pd.read_pickle(output_path + "doc_full2.pkl")


df_full_ml = dici_full["ml"]
df_full_standard = dici_full["standard"]
df_full_ext = dici_full["ext"]

df_est_ml = dici_est["ml"]
df_est_standard = dici_est["standard"]
df_est_ext = dici_est["ext"]

# Making one dataframe with different columns for different prediction types
def make_ana_df(real_dici, predicted_dici):
    df_real = real_dici["ml"]
    df_predicted_ml = predicted_dici["ml"]
    df_predicted_standard = predicted_dici["standard"]
    df_predicted_ext = predicted_dici["ext"]

    # These are the real people, not the imputed ones
    relevant = df_real[df_real["predicted"]==0]

    # Adding columns with predictions by normal ML algorithm
    together = pd.merge(relevant, df_predicted_ml,
                        on=["pid", "year"], suffixes=["_real", "_ml"])

    # Adding columns with predictions by Logit/OLS
    together = pd.merge(together, df_predicted_standard,
                        on=["pid", "year"], suffixes=["", "_standard"])

    # Adding columns with predictions by extended ML algorithm
    together = pd.merge(together, df_predicted_ext,
                        on=["pid", "year"], suffixes=["", "_ext"])

    # Specifying how far into the future the prediction goes
    together["period_ahead"] = np.nan
    together["max_period"] = np.nan
    together["max_period"] = together.groupby("pid")["year"].transform("last")
    together["period_ahead"] = together["max_period"] - together["year"]
    together.drop("max_period", axis=1, inplace=True)

    return together

# Making a cohort depending on supplied birthyears
def make_cohort(dataf, birthyears):
    dataf = dataf.copy()

    birthyear = dataf["year"] - dataf["age"]
    condition = [by in birthyears for by in birthyear]
    dataf = dataf.loc[condition]
    # Only using data from western Germany for now
    dataf = dataf[dataf["east"]==0]

    dataf_renamed = dataf.iloc[:,76:113].add_suffix("_standard")
    dataf = pd.concat([dataf.iloc[:,:76], dataf_renamed, dataf.iloc[:,113:]], axis=1)

    return dataf

# Only keeping those, who are in the sample in the very first year
def first_year(dataf):
    a = (dataf.groupby("pid")["year"].min()).to_frame()
    a.reset_index(inplace=True)
    b = a.loc[a["year"]==min(a["year"]), "pid"].tolist()

    c = [pid in b for pid in dataf["pid"]]
    dataf = dataf[c]
    return dataf

###############################################################################
def run_rightshape(current_week):
    # Specifying week for folder selection
    output_week = output_path + "week" + current_week + "/"
    pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)


    # Running functions
    df_analysis = make_ana_df(dici_full, dici_est)

    cohorts = np.arange(1945, 1955)
    df_out = make_cohort(df_analysis, cohorts)

    #df_out = df_out[(df_out["age_real"]<60)&(df_out["age_real"]>29)]
    #df_out = first_year(df_out)
    df_out.to_pickle(output_week + "df_analysis_full")

###############################################################################
if __name__ == "__main__":
    # Specifying week for folder selection
    current_week = "46"
    output_week = output_path + "week" + current_week + "/"
    pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)


    # Running functions
    df_analysis = make_ana_df(dici_full, dici_est)

    cohorts = np.arange(1945, 1955)
    df_out = make_cohort(df_analysis, cohorts)

    #df_out = df_out[(df_out["age_real"]<60)&(df_out["age_real"]>29)]
    #df_out = first_year(df_out)
    df_out.to_pickle(output_week + "df_analysis_full")
