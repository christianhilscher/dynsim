from pathlib import Path
import numpy as np
import pandas as pd
import pickle

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, LinearRegression

from standard import getdf, get_dependent_var

###############################################################################
dir = Path(__file__).resolve().parents[2]
input_path = dir / "input"
model_path = dir / "src/estimation/models/"
###############################################################################


def data_general(dataf, dep_var, estimate=1):
    dataf = dataf.copy()


    if estimate == 1:
        dataf.rename(columns={dep_var: 'dep_var'}, inplace=True)
    else:
        dataf.drop(dep_var, axis=1, inplace=True)
        dataf.drop('personweight', axis=1, inplace=True)

    if dep_var == "employment_status":
        vars_drop = ["pid",
                     "hid",
                     "orighid",
                     "age_max",
                     "predicted",
                     "hhweigth",
                     "retired",
                     "working",
                     "fulltime",
                     "hours",
                     "gross_earnings"]
    elif dep_var == "hours":
        vars_drop = ["pid",
                     "hid",
                     "orighid",
                     "age_max",
                     "predicted",
                     "hhweigth",
                     "gross_earnings"]
    else:
        vars_drop = ["pid",
                     "hid",
                     "orighid",
                     "age_max",
                     "predicted",
                     "hhweigth"]

    for var in vars_drop:
        if var in dataf.columns.tolist():
            dataf.drop(var, axis=1, inplace=True)
        else:
            pass

    return dataf

def _prepare_classifier(dataf):
    dataf = dataf.copy()

    y = dataf['dep_var']
    X = dataf.drop('dep_var', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

    # Making weights
    weights_train = X_train['personweight']
    X_train.drop('personweight', axis=1, inplace=True)


    weights_test = X_test['personweight']
    X_test.drop('personweight', axis=1, inplace=True)


    if "personweight_interacted" in X.columns.tolist():
        X_train.drop('personweight_interacted', axis=1, inplace=True)
        X_test.drop('personweight_interacted', axis=1, inplace=True)
    else:
        pass

    # Scaling
    X_train_scaled = StandardScaler().fit_transform(np.asarray(X_train))
    X_test_scaler = StandardScaler().fit(np.asarray(X_test))
    X_test_scaled = X_test_scaler.transform(np.asarray(X_test))

    # Coeffs feature_names
    feature_names = X_train.columns.tolist()

    # For Standard Part:
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

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
                'weights': weights_train,
                'X_scaler': X_test_scaler}
    return out_dici

def _prepare_regressor(dataf, dep_var):
    dataf = dataf.copy()

    y = dataf['dep_var']
    X = dataf.drop('dep_var', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

    # Making weights
    weights_train = X_train['personweight']
    X_train.drop('personweight', axis=1, inplace=True)

    weights_test = X_test['personweight']
    X_test.drop('personweight', axis=1, inplace=True)

    # Scaling
    X_train_scaled = StandardScaler().fit_transform(np.asarray(X_train))
    X_test_scaler = StandardScaler().fit(np.asarray(X_test))
    X_test_scaled = X_test_scaler.transform(np.asarray(X_test))
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


    out_dici = {'y_scaler': y_test_scaler,
                'X_scaler': X_test_scaler,
                'lgb_train': lgb_train,
                'lgb_test': lgb_test,
                'features': feature_names,
                'weights': weights_train}
    pickle.dump(y_test_scaler,
                open(model_path / str(dep_var + "_scaler_ext"), 'wb'))
    return out_dici

def _estimate(dataf, dep_var, type):
    dataf = dataf.copy()

    dataf = data_general(dataf, dep_var)
    dataf.dropna(inplace=True)

    if type == 'regression':
        dict = _prepare_regressor(dataf, dep_var)
        params = {'boosting_type' : 'gbdt',
                  'n_estimators': 350,
                  'objective' : 'l2',
                  'metric' : 'l2',
                  'num_leaves' : 31,
                  'learning_rate' : 0.15,
                  'feature_fraction': [0.9],
                  'bagging_fraction': [0.8],
                  'bagging_freq': [5],
                  'verbose' : 5,
                  'early_stopping_rounds': 5}
        pickle.dump(dict['y_scaler'],
                    open(model_path / str(dep_var + "_y_scaler_multi"), 'wb'))
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
                'verbose': 0,
                'early_stopping_rounds': 5}
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
                  'verbose': 0,
                  'early_stopping_rounds': 5}

    modl = lgb.cv(params,
                     train_set = dict['lgb_train'],
                     nfold=10,
                     return_cvbooster=True)

    # Make directory if it doesn't exist yet
    # Path(model_path / dep_var).mkdir(parents=True, exist_ok=True)
    # modl.save_model(str(model_path / dep_var / "_extended.txt"))

    # pickle.dump(dict['X_scaler'],
    #             open(model_path / dep_var / "_X_scaler_multi", 'wb'))
    return modl




###############################################################################
if __name__ == "__main__":
    df = pd.read_pickle(input_path / 'merged').dropna()
    df1 = getdf(df)

    _estimate(df1, "employment_status", "multiclass")
    # _estimate(df1, "hours", "regression")
    # _estimate(df1, "gross_earnings", "regression")


df = pd.read_pickle(input_path / 'merged').dropna()
df1 = getdf(df)

abc = _estimate(df1, "employment_status", "multiclass")

abc["cvbooster"].boosters[1].feature_importance()