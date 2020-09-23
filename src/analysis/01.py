import numpy as np
import pandas as pd
import pathlib
import os


###############################################################################
current_week = "38"
output_week = "/Users/christianhilscher/desktop/dynsim/output/week" + str(current_week) + "/"
pathlib.Path(output_week).mkdir(parents=True, exist_ok=True)
###############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
output_path = "/Users/christianhilscher/Desktop/dynsim/output/"

dici_full = pd.read_pickle(output_path + "doc_full.pkl")
dici_est = pd.read_pickle(output_path + "doc_full2.pkl")


df_full_ml = dici_full["ml"]
df_full_standard = dici_full["standard"]
df_full_ext = dici_full["ext"]

df_est_ml = dici_est["ml"]
df_est_standard = dici_est["standard"]
df_est_ext = dici_est["ext"]

def make_ana_df(real_dici, predicted_dici):
    df_real = real_dici["ml"]
    df_predicted_ml = predicted_dici["ml"]
    df_predicted_standard = predicted_dici["standard"]
    df_predicted_ext = predicted_dici["ext"]

    relevant = df_real[df_real["predicted"]==0]

    together = pd.merge(relevant, df_predicted_ml,
                        on=["pid", "year"], suffixes=["_real", "_ml"])
    together = pd.merge(together, df_predicted_standard,
                        on=["pid", "year"], suffixes=["", "_standard"])
    together = pd.merge(together, df_predicted_ext,
                        on=["pid", "year"], suffixes=["", "_ext"])

    together.sort_values(by=["pid", "year"])
    together["period_ahead"] = 0

    out = pd.DataFrame()
    tmp = together
    j=0

    while len(tmp)>0:
        j+=1
        cond = tmp["pid"].duplicated(keep="first")
        not_dups = tmp[~cond]
        dups = tmp[cond]

        out = pd.concat([out, not_dups])
        dups["period_ahead"] += 1
        print("Done with period", j)
        tmp = dups.sort_values(by=["pid", "year"])

    return out

def make_cohort(dataf, birthyears):
    dataf = dataf.copy()

    birthyear = dataf["year"] - dataf["age"]
    condition = [by in birthyears for by in birthyear]
    dataf = dataf.loc[condition]
    dataf = dataf[dataf["east"]==0]

    dataf_renamed = dataf.iloc[:,76:113].add_suffix("_standard")
    dataf = pd.concat([dataf.iloc[:,:76], dataf_renamed, dataf.iloc[:,113:]], axis=1)

    return dataf


df_analysis = make_ana_df(dici_full, dici_est)

cohorts = np.arange(1945, 1955)
df_out = make_cohort(df_analysis, cohorts)


df_out = df_out[(df_out["age_real"]<60)&(df_out["age_real"]>29)]
df_out.to_pickle(output_week + "df_analysis_full")
