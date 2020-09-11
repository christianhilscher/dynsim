import numpy as np
import pandas as pd
import os
import pickle

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
dir = "/Users/christianhilscher/Desktop/dynsim"
input_path = dir + "/input/"
working_dir = dir + "/src/data_preparation"

os.chdir(working_dir)
from cleaning import SOEP_to_df
from data_prep import SOEP_to_df_old


def make_hh_vars(dataf):
    """
    Generating variables which belong to one household such as HH-income
    """
    dataf = dataf.copy()
    dataf = _get_multiindex(dataf)
    dataf = _indicate_birth(dataf)
    dataf.reset_index(inplace=True, drop=True)
    return dataf


def _get_tupleindices(dataf):
    years = dataf['year'].tolist()
    hids = dataf['hid'].tolist()
    return list(zip(years, hids))

def _get_multiindex(dataf):
    dataf = dataf.copy()
    index_list = _get_tupleindices(dataf)

    mindex = pd.MultiIndex.from_tuples(index_list, names=['year' ,
                                                          'hid'])
    dataf_out = dataf.set_index(mindex)
    dataf_out = dataf_out.sort_index(level=1)

    return dataf_out

def _indicate_birth(dataf):
    """
    Indictaes whether a mother has had a baby in that particular year
    """

    dataf = dataf.copy()

    minage = dataf.groupby(level=['year', 'hid'])['age'].min()
    dataf["minage"] = minage
    dataf["birth"] = 0
    dataf.loc[(dataf["minage"]==0)&(dataf["female"]==1)&(dataf["child"]==0), "birth"] = 1
    dataf.drop("minage", axis=1, inplace=True)

    return dataf




# Making new dataset from original SOEP
df_pgen = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/pgen.dta")
df_hgen = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/hgen.dta")
df_ppathl = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/ppathl.dta")
df_hpathl = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/hpathl.dta")
df_hbrutto = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/hbrutto.dta")
df_pkal = pd.read_stata("/Volumes/B/soep.v35/STATA_DEEN_v35/Stata/pkal.dta")


df_pgen = df_pgen[["hid", "pid" , "syear", "pglabgro", "pgemplst", "pglfs", "pgtatzeit", "pgerwzeit", "pgpsbil", "pgfamstd"]]
df_hgen = df_hgen[["hid", "syear", "hgheat", "hgrent", "hgtyp1hh"]]


df_ppathl = df_ppathl[["hid", "pid", "syear", "sex", "gebjahr", "migback", "phrf"]]
df_hpathl = df_hpathl[["hid", "syear", "hhrf"]]
df_hbrutto = df_hbrutto[["hid", "syear", "bula"]]
df_pkal = df_pkal[["pid", "syear", "kal1e01"]]

# Merging datasets from SOEP
person_df = pd.merge(df_pgen, df_ppathl, on=["pid", "syear"], how="left")
person_df = pd.merge(person_df, df_pkal, on=["pid", "syear"], how="left")
hh_df = pd.merge(df_hgen, df_hpathl, on=["hid", "syear"], how="left")
hh_df = pd.merge(hh_df, df_hbrutto, on=["hid", "syear"], how="left")

person_df.drop("hid_y", axis=1, inplace=True)
person_df.rename(columns={"hid_x": "hid"}, inplace=True)

full_df = pd.merge(person_df, hh_df, on=["hid", "syear"])
full_df.columns.tolist()

try1 = SOEP_to_df(full_df)
try1.drop("tenure", axis=1, inplace=True)
try1.drop("heizkosten", axis=1, inplace=True)
try1.drop("bruttokaltmiete", axis=1, inplace=True)

names_list = try1.columns.tolist()

# Reading in old dataset
old_df = pd.read_pickle("/Users/christianhilscher/Desktop/dynsim/input/old/full")
orig_df = SOEP_to_df_old(old_df)


finish = pd.merge(try1, orig_df, on=["pid", "year"], how="outer")
orig_df.shape

names_list.remove("pid")
names_list.remove("year")
for name in names_list:
    finish[name] = finish[name+"_x"]
    finish[name].fillna(finish[name+"_y"], inplace=True)

names_list.append("year")
names_list.append("pid")
names_list
finish_small = finish[names_list]
finish_final = make_hh_vars(finish_small)
finish_final = finish_final[finish_final["age"]<99]
finish_final.dropna().to_pickle(input_path + "merged")


cond = [age in np.arange(16, 66) for age in finish_final["age"]]
finish_final[cond].dropna().to_pickle(input_path + "workingage")

sum(full_df["pglfs"] == "[6] NW-unemployed")

full_df.loc[full_df["pglfs"]=="[6] NW-unemployed" , "syear"] = 3000


sum(finish_final["employment_status"]==4)
