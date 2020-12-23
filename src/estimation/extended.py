from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import lightgbm as lgb

###############################################################################
dir = Path(__file__).parents[2]
input_path = dir / "input"
model_path = dir /"src/estimation/models/"
###############################################################################

def getdf(dataf):
    dataf = dataf.copy()

    condition = dataf.groupby('pid')['year'].count()>2
    dataf = dataf.set_index('pid')[condition]
    year_list = dataf['year'].unique()

    dataf['hours_t1'] = np.NaN
    dataf['gross_earnings_t1'] = np.NaN

    dataf_out = pd.DataFrame()
    for i in year_list[2:]:
        df_now = dataf[dataf['year'] == i].copy()
        df_yesterday = dataf[dataf['year'] == (i-1)].copy()
        df_twoyesterdays = dataf[dataf['year'] == (i-2)].copy()

        df_now['lfs_t1'] = df_yesterday['lfs']
        df_now['working_t1'] = df_yesterday['working']
        df_now['fulltime_t1'] = df_yesterday['fulltime']
        df_now['hours_t1'] = df_yesterday['hours']
        df_now['hours_t2'] = df_twoyesterdays['hours']
        df_now['gross_earnings_t1'] = df_yesterday['gross_earnings']
        df_now['gross_earnings_t2'] = df_twoyesterdays['gross_earnings']

        dataf_out = pd.concat([dataf_out, df_now])

    dataf_out.reset_index(inplace=True)
    dataf_out.dropna(inplace=True)
    return dataf_out

def get_dependent_var(dataf, dep_var):
    dataf = dataf.copy()

    dataf.rename(columns={dep_var: 'dep_var'}, inplace=True)
    return dataf

def _prepare_classifier(dataf):
    dataf = dataf.copy()

    y = dataf['dep_var']
    X = dataf.drop('dep_var', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

    # Making weights
    weights_train = X_train['personweight']
    #X_train.drop('personweight', axis=1, inplace=True)

    weights_test = X_test['personweight']
    #X_test.drop('personweight', axis=1, inplace=True)


    if "personweight_interacted" in X.columns.tolist():
        X_train.drop('personweight_interacted', axis=1, inplace=True)
        X_test.drop('personweight_interacted', axis=1, inplace=True)
    else:
        pass

    # Scaling
    X_train_scaled = StandardScaler().fit_transform(np.asarray(X_train))
    X_test_scaled = StandardScaler().fit_transform(np.asarray(X_test))

    # Coeffs feature_names
    feature_names = X_train.columns.tolist()

    # For ML part:
    lgb_train = lgb.Dataset(X_train_scaled, y_train,
                            weight = weights_train)
    lgb_test = lgb.Dataset(X_test_scaled, y_test,
                           weight = weights_test)

    out_dici = {'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'lgb_train': lgb_train,
                'lgb_test': lgb_test,
                'features': feature_names,
                'weights': weights_train}
    return out_dici


def _prepare_regressor(dataf):
    dataf = dataf.copy()

    y = dataf['dep_var']
    X = dataf.drop('dep_var', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

    # Making weights
    weights_train = X_train['personweight']
    #X_train.drop('personweight', axis=1, inplace=True)

    weights_test = X_test['personweight']
    #X_test.drop('personweight', axis=1, inplace=True)

    # Scaling
    X_train_scaled = StandardScaler().fit_transform(np.asarray(X_train))
    X_test_scaled = StandardScaler().fit_transform(np.asarray(X_test))
    y_train_scaled = StandardScaler().fit_transform(np.asarray(y_train).reshape(-1,1))

    # Saving the scaler of the test data to convert the predicted values again
    y_test_scaler = StandardScaler().fit(np.asarray(y_test).reshape(-1,1))
    y_test_scaled = y_test_scaler.transform(np.asarray(y_test).reshape(-1,1))

    feature_names = X_train.columns.tolist()
    y_test_scaled = np.ravel(y_test_scaled)
    y_train_scaled = np.ravel(y_train_scaled)

    # For ML part:
    lgb_train = lgb.Dataset(X_train_scaled, y_train,
                            weight = weights_train)
    lgb_test = lgb.Dataset(X_test_scaled, y_test,
                           weight = weights_test)


    out_dici = {'X_train': X_train_scaled,
                'X_test': X_test,
                'y_train': y_train_scaled,
                'y_test': y_test,
                'scaler': y_test_scaler,
                'lgb_train': lgb_train,
                'lgb_test': lgb_test,
                'features': feature_names,
                'weights': weights_train}
    return out_dici

#############################################################################

def data_general(dataf, dep_var, estimate=1):
    dataf = dataf.copy()

    if estimate == 1:
        dataf = get_dependent_var(dataf, dep_var)
    else:
        dataf = get_dependent_var(dataf, dep_var)
        dataf.drop('dep_var', axis=1, inplace=True)

    vars_drop = ['pid',
                 'hid',
                 'orighid',
                 'age_max',
                 'predicted']

    for var in vars_drop:
        if var in dataf.columns.tolist():
            dataf.drop(var, axis=1, inplace=True)
        else:
            pass

    return dataf

def _estimate(dataf, dep_var, type):
    dataf = dataf.copy()

    dataf = data_general(dataf, dep_var)
    dataf.dropna(inplace=True)

    if type == 'regression':
        dict = _prepare_regressor(dataf)
        params = {'boosting_type' : 'gbdt',
                  'n_estimators': 350,
                  'objective' : 'l2',
                  'metric' : 'l2',
                  'num_leaves' : 31,
                  'learning_rate' : 0.15,
                  'feature_fraction': [0.9],
                  'bagging_fraction': [0.8],
                  'bagging_freq': [5],
                  'verbose' : 5}
    elif type == 'binary':
            dict = _prepare_classifier(dataf)
            params = {'task' : 'train',
                'boosting_type' : 'gbdt',
                'n_estimators': 350,
                'objective': 'binary',
                'eval_metric': 'logloss',
                'learning_rate': 0.05,
                'feature_fraction': [0.9],
                'num_leaves': 31,
                'verbose': 0}
    else:
        dict = _prepare_classifier(dataf)
        params = {'task' : 'train',
                  'boosting_type' : 'gbdt',
                  'n_estimators': 350,
                  'objective': 'multiclass',
                  'num_class': len(dict['y_train'].unique()),
                  'eval_metric': 'multi_logloss',
                  'learning_rate': 0.05,
                  'feature_fraction': [0.9],
                  'num_leaves': 31,
                  'verbose': 0}

    modl = lgb.train(params,
                     train_set = dict['lgb_train'],
                     valid_sets = dict['lgb_test'],
                     feature_name = dict['features'])

    if dep_var == 'gross_earnings':
        modl.save_model(str(model_path / "earnings_extended.txt"))
    else:
        modl.save_model(str(model_path / dep_var / "_extended.txt"))


# df = pd.read_pickle(input_path + 'imputed08').dropna()
# df1 = getdf(df)
#
# _estimate(df1, 'lfs', 'binary')
# _estimate(df1, 'working', 'binary')
# _estimate(df1, 'fulltime', 'binary')
# _estimate(df1, 'hours', 'regression')
# _estimate(df1, 'gross_earnings', 'regression')
# _estimate(df1, 'birth', 'binary')
# _estimate(df1, 'employment_status', 'multiclass')
