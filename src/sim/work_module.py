import numpy as np
import pandas as pd
import pickle
import os

import lightgbm as lgb
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
##############################################################################
##############################################################################
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"
model_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/models/"
estimation_path = "/Users/christianhilscher/desktop/dynsim/src/estimation/"
sim_path = "/Users/christianhilscher/desktop/dynsim/src/sim/"

os.chdir(estimation_path)
from standard import getdf, data_retired, data_working, data_fulltime, data_hours, data_earnings
from multic import data_general
os.chdir(sim_path)
##############################################################################
##############################################################################

# Functions used for predicting values
def scale_data(dataf, dep_var=None, multi=0):
    dataf = dataf.copy()
    if multi == 1:

        scaler = pd.read_pickle(model_path + dep_var + "_X_scaler_multi")
    else:
        scaler = pd.read_pickle(model_path + dep_var + "_X_scaler")
    X = scaler.transform(np.asarray(dataf))
    return X

def _logit(X, variable):
    X = X.copy()

    X_scaled = scale_data(X, variable)
    #X_scaled['const'] = 1
    estimator  = pd.read_pickle(model_path + variable + "_logit")
    pred = estimator.predict(X_scaled)

    pred_scaled = np.zeros(len(pred))
    if np.any(pred)>0.5:
        pred_scaled[pred>0.5] = 1
    else:
        raise ValueError('Only Zeros')

    return pred_scaled

def _ols(X, variable):
    X= X.copy()

    X_scaled = scale_data(X, variable)
    #X['const'] = 1
    estimator  = pd.read_pickle(model_path + variable + "_ols")
    pred = estimator.predict(X_scaled)

    scaler = pd.read_pickle(model_path + variable + "_y_scaler")
    pred_scaled = scaler.inverse_transform(pred)
    pred_scaled[pred_scaled<0] = 0

    return pred_scaled

def _ml(X, variable):
    X = X.copy()

    X_scaled = scale_data(X, variable)
    estimator = lgb.Booster(model_file = model_path + variable + '_ml.txt')
    pred = estimator.predict(X_scaled)

    if variable in ['hours', 'gross_earnings']:
        # pred_scaled = pred
        # Inverse transform regression results
        scaler = pd.read_pickle(model_path + variable + "_y_scaler")
        pred_scaled = scaler.inverse_transform(pred)
    else:
        # Make binary prediction to straight 0 and 1
        pred_scaled = np.zeros(len(pred))
        pred_scaled[pred>0.5] = 1

    pred_scaled[pred_scaled<0] = 0
    return pred_scaled

def _ext(X, variable):
    X = X.copy()
    X.reset_index(drop=True, inplace=True)
    X_scaled = scale_data(X, variable, multi=1)
    estimator = lgb.Booster(model_file = model_path + \
                            variable + '_extended.txt')
    pred = estimator.predict(X_scaled)

    if variable == "employment_status":
        weighted_res = get_results(X, pred, 0)
        predictions = draw_status(weighted_res)

    else:
        predictions = pred
        # scaler = pd.read_pickle(model_path + variable + "_scaler_ext")
        # predictions = scaler.inverse_transform(pred)

    predictions[predictions<0] = 0
    return predictions
###############################################################################
# Making household wide variables
def make_hh_vars(dataf):
    dataf = dataf.copy()
    dataf.set_index(['hid'],inplace=True)

    dataf = _hh_income(dataf)
    dataf = _hh_age_youngest(dataf)
    dataf = _hh_children(dataf)
    dataf = _hh_size(dataf)
    dataf = _hh_fraction_working(dataf)
    dataf.reset_index(inplace=True)
    return dataf

def _hh_income(dataf):
    dataf = dataf.copy()
    earnings = dataf.groupby('hid')['gross_earnings_t1'].sum()
    dataf['hh_income'] = earnings
    return dataf

def _hh_size(dataf):
    dataf = dataf.copy()
    size = dataf.groupby('hid')['gross_earnings'].size()
    dataf['n_people'] = size
    return dataf

def _hh_children(dataf):
    dataf = dataf.copy()
    children = dataf.groupby('hid')['child'].sum()
    dataf['n_children'] = children
    return dataf

def _hh_fraction_working(dataf):
    dataf = dataf.copy()

    dataf = _hh_size(dataf)
    dataf = _hh_children(dataf)

    total = dataf.groupby('hid')['working_t1'].sum()
    dataf['total_working'] = total

    dataf['n_adults'] = dataf['n_people'] - dataf['n_children']
    if dataf['n_adults'].any() == 0:
        raise ValueError('No adult in HH')
    else:
        dataf['hh_frac_working'] = dataf['total_working']/dataf['n_adults']

    # Children could also be working, but bound it at 1
    dataf.loc[dataf["hh_frac_working"]>1, "hh_frac_working"] = 1

    # NaN cannot occur
    dataf.loc[dataf["hh_frac_working"].isna(), "hh_frac_working"] = 0

    dataf.drop(['total_working', 'n_adults'], axis=1, inplace=True)
    return dataf

def _hh_age_youngest(dataf):
    dataf = dataf.copy()

    smallest_age = dataf.groupby('hid')['age'].min()
    dataf['hh_youngest_age'] = smallest_age
    return dataf


##############################################################################

def sim_retired(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_retired(dataf, estimate=0)
        predictions = _logit(X, 'retired')
    elif type == 'ml':
        X = data_retired(dataf, estimate=0)
        predictions = _ml(X, 'retired')
    elif type == "ext":
        predictions = np.zeros(len(dataf))
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_working(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_working(dataf, estimate=0)
        predictions = _logit(X, 'working')
    elif type == 'ml':
        X = data_working(dataf, estimate=0)
        predictions = _ml(X, 'working')
    elif type == "ext":
        predictions = np.zeros(len(dataf))
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_fulltime(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_fulltime(dataf, estimate=0)
        predictions = _logit(X, 'fulltime')
    elif type == 'ml':
        X = data_fulltime(dataf, estimate=0)
        predictions = _ml(X, 'fulltime')
    elif type == "ext":
        predictions = np.zeros(len(dataf))
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_hours(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_hours(dataf, estimate=0)
        predictions = _ols(X, 'hours')
    elif type == 'ml':
        X = data_hours(dataf, estimate=0)
        predictions = _ml(X, 'hours')
    elif type == "ext":
        X = data_general(dataf, "hours", estimate=0)
        predictions = _ext(X, "hours")
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_earnings(dataf, type):
    dataf = dataf.copy()

    if type == 'standard':
        X = data_earnings(dataf, estimate=0)
        predictions = _ols(X, 'gross_earnings')
    elif type == 'ml':
        X = data_earnings(dataf, estimate=0)
        predictions = _ml(X, 'gross_earnings')
    elif type == "ext":
        X = data_general(dataf, "gross_earnings", estimate=0)
        predictions = _ext(X, "gross_earnings")
    else:
        raise ValueError("Unkown Type")

    return predictions

def sim_multi_employment(dataf):
    dataf = dataf.copy()

    X = data_general(dataf, "employment_status", estimate=0)
    predictions = _ext(X, "employment_status")

    return predictions


def to_category(dataf):
    dataf = dataf.copy()

    ne = (dataf["working"]==0) & (dataf["retired"] == 0)
    teilzeit = (dataf["working"]==1)&(dataf["fulltime"]==0)
    vollzeit = (dataf["working"]==1)&(dataf["fulltime"]==1)

    dataf.loc[ne, "employment_status"] = 0
    dataf.loc[dataf["retired"]==1, "employment_status"] = 1
    dataf.loc[teilzeit, "employment_status"] = 2
    dataf.loc[vollzeit, "employment_status"] = 3

    return dataf

def to_binary(dataf):
    dataf = dataf.copy()

    ne = dataf["employment_status"] == 0
    dataf.loc[ne, "working"] = 0
    dataf.loc[ne, "retired"] = 0
    dataf.loc[ne, "fulltime"] = 0

    rente = dataf["employment_status"] == 1
    dataf.loc[rente, "working"] = 0
    dataf.loc[rente, "fulltime"] = 0
    dataf.loc[rente, "retired"] = 1

    teilzeit = dataf["employment_status"] == 2
    dataf.loc[teilzeit, "working"] = 1
    dataf.loc[teilzeit, "fulltime"] = 0
    dataf.loc[teilzeit, "retired"] = 0

    vollzeit = dataf["employment_status"] == 3
    dataf.loc[vollzeit, "working"] = 1
    dataf.loc[vollzeit, "fulltime"] = 1
    dataf.loc[vollzeit, "retired"] = 0

    return dataf

# Functions for transition matrices
def read_transition_data():
    trans_matrices = pd.read_pickle(estimation_path + "transition_matrices/full_sample")
    return trans_matrices

def get_access(dataf):

    dataf["sex"] = "female"
    dataf.loc[dataf["female"]==0, "sex"] = "male"

    bins = np.arange(0, 101, 5)
    dataf["age_bin"] = pd.cut(dataf["age"], bins)
    dataf["left"] = [str(bin.left) for bin in dataf["age_bin"]]

    dataf["name"] = list(zip(dataf["sex"], dataf["left"]))
    dataf["name"] = dataf["name"].apply(get_name)

    dataf.drop("sex", axis=1, inplace=True)
    dataf.drop("age_bin", axis=1, inplace=True)
    dataf.drop("left", axis=1, inplace=True)

    return dataf

def get_name(itr):
    seq = (itr[0], itr[1])
    sep = "_"
    return sep.join(seq)

def get_cond_prob(age_interval, employment_status, dici):
    probs = dici[age_interval].iloc[int(employment_status),0:4]
    marg_prob = dici[age_interval].iloc[int(employment_status),-1]

    if marg_prob != 0:
        out = probs/marg_prob
    else:
        out = np.zeros(4)
    return out

def weighting(prediction, transition, share_prediction):
    weighted = np.empty_like(prediction)
    out = np.empty(len(prediction))

    weighted = share_prediction * prediction + (1-share_prediction) * transition
    return weighted

def get_results(dataf, predictions, share):
    dataf = dataf.copy()

    transition_matrices = read_transition_data()

    # Adapt dataframe to compare probabilities
    dataf = get_access(dataf)
    dataf.reset_index(drop=True, inplace=True)

    assert(
        dataf.isnull().sum().sum() == 0
    ), ValueError("You have missings up here")

    own_prob = np.zeros_like(predictions)
    i = 0
    for (name, empl) in zip(dataf["name"], dataf["employment_status_t1"]):
        own_prob[i] = get_cond_prob(name,
                                    empl,
                                    transition_matrices)
        i += 1
    assert(
        sum(sum(np.isnan(own_prob))) == 0
    ), ValueError("You have missings up here")
    # Do the weighting
    out = weighting(predictions, own_prob, share)
    return out

def draw_status(weighted_results):

    # Adding zero for intervals later
    withzeros = np.zeros((weighted_results.shape[0],
                          weighted_results.shape[1] + 1))

    withzeros[:, 1:] = weighted_results

    # cumulate and make intervals
    cumulated = [withzeros[i,:].cumsum() for i in np.arange(len(withzeros))]
    intervs = [pd.arrays.IntervalArray.from_breaks(i) for i in cumulated]

    # random draw
    draw = np.random.uniform(size=withzeros.shape[0])

    # depending on rnaomd draw assign status
    status = np.empty(len(draw))
    for stat in np.arange(4):

        cond = [draw[i] in intervs[i][stat] for i in np.arange(len(draw))]
        status[cond] = stat

    return status
