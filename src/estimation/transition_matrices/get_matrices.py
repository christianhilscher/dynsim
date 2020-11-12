import numpy as np
import pandas as pd
import pickle
import os, pathlib

##############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
##############################################################################
# Getting dataframe into right shape
def getdf(dataf):
    dataf = dataf.copy()

    # Only keeping those with more than two consective years
    condition = dataf.groupby('pid')['year'].count()>2
    dataf = dataf.set_index('pid')[condition]
    year_list = dataf['year'].unique()

    # Making space
    dataf['hours_t1'] = np.NaN
    dataf['gross_earnings_t1'] = np.NaN

    # Final dataframe for output
    dataf_out = pd.DataFrame()

    # For each year use the previous year's values to fill up t-1 and t-2 columns
    for i in np.sort(year_list)[2:]:
        df_now = dataf[dataf['year'] == i].copy()
        df_yesterday = dataf[dataf['year'] == (i-1)].copy()
        df_twoyesterdays = dataf[dataf['year'] == (i-2)].copy()

        df_now['retired_t1'] = df_yesterday['retired']
        df_now['working_t1'] = df_yesterday['working']
        df_now['fulltime_t1'] = df_yesterday['fulltime']
        df_now['hours_t1'] = df_yesterday['hours']
        df_now['hours_t2'] = df_twoyesterdays['hours']
        df_now['gross_earnings_t1'] = df_yesterday['gross_earnings']
        df_now['gross_earnings_t2'] = df_twoyesterdays['gross_earnings']
        df_now['employment_status_t1'] = df_yesterday['employment_status']
        df_now['employment_status_t2'] = df_twoyesterdays['employment_status']

        dataf_out = pd.concat([dataf_out, df_now])

    dataf_out.reset_index(inplace=True)
    dataf_out.dropna(inplace=True)
    return dataf_out

def transition_probability(dataf, value_t, value_t1):

    return (sum((dataf["employment_status"]==value_t) &
                (dataf["employment_status_t1"]==value_t1))/len(dataf))

def marginal_probability(dataf, value_t, t=True):

    if t==True:
        return (sum((dataf["employment_status"]==value_t))/len(dataf))
    else:
        return (sum((dataf["employment_status_t1"]==value_t))/len(dataf))

def make_matrix(dataf):
    dataf = dataf.copy()

    n_stati = 4
    trans_mat = np.zeros((n_stati + 1, n_stati + 1))

    # Probabilities
    for i in np.arange(n_stati):
        for j in np.arange(n_stati):
            trans_mat[i, j] = transition_probability(dataf, i, j)

    # Marginal probabilities
    for i in np.arange(n_stati):
            trans_mat[i, -1] = marginal_probability(dataf, i, t=True)

    for j in np.arange(n_stati):
            trans_mat[-1, j] = marginal_probability(dataf, j, t=False)

    assert(
        np.round(sum(trans_mat[-1,]), decimals=2) == np.round(sum(trans_mat[:,-1]), decimals=2) == 1
    ), ValueError("Marginal probabilities don't aff up to 1.")
    trans_mat[-1, -1] = sum(trans_mat[-1,])


    return pd.DataFrame(trans_mat)

def write_matrices(dataf):
    dataf = dataf.copy()

    bins = np.arange(0, 101, 1)
    dataf["age_bin"] = pd.cut(dataf["age"], bins, right=False)

    dataf.sort_values("age", inplace=True)

    dici = {}

    for bin in dataf["age_bin"].unique():
        print("Age bracket starting with ", bin.left)

        for sex in dataf["female"].unique():

            if sex == 1:
                df_tmp = dataf[(dataf["age_bin"] == bin)&(dataf["female"]==1)]
                title = "female_" + str(bin.left)
            else:
                df_tmp = dataf[(dataf["age_bin"] == bin)&(dataf["female"]==0)]
                title = "male_" + str(bin.left)


            dici[title] = make_matrix(df_tmp)
    return dici

# Now getting transition probabilities for our cohort
def make_cohort(dataf, birthyears):
    dataf = dataf.copy()

    birthyear = dataf["year"] - dataf["age"]
    condition = [by in birthyears for by in birthyear]
    dataf = dataf.loc[condition]
    dataf = dataf[dataf["east"]==0]

    return dataf


###############################################################################
if __name__ == "__main__":

    # Read in data
    df = pd.read_pickle(input_path + "merged")
    df1 = getdf(df)
    # Get matrices for whole sample
    trans_matrices = write_matrices(df1)

    # Now only for our cohort
    cohorts = np.arange(1945, 1955)
    df_cohort = make_cohort(df1, cohorts)

    trans_matrices_cohort = write_matrices(df_cohort)

    # Dump the in same folder to overwrite values
    trans_matrices.update(trans_matrices_cohort)

    # Saving final dictionary
    os.chdir(estimation_path)
    pickle.dump(trans_matrices,
                open("transition_matrices/full_sample", "wb"))
