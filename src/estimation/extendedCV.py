from pathlib import Path
import numpy as np
import pandas as pd
import pickle

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, LinearRegression

from standard import getdf, get_dependent_var
from extended import data_general, prepare_classifier, prepare_regressor

###############################################################################
# dir = Path(__file__).resolve().parents[2]
dir = Path("/home/christian/dynasim/")
input_path = dir / "input"
model_path = dir / "src/estimation/models/"
###############################################################################


def _estimate(dataf, dep_var, type):
    dataf = dataf.copy()

    dataf = data_general(dataf, dep_var)
    dataf.dropna(inplace=True)

    if type == 'regression':
        dici = prepare_regressor(dataf, dep_var)
        # pickle.dump(dici['y_scaler'],
        #             open(model_path / str(dep_var + "_y_scaler_multi"), 'wb'))
        
        estimator = lgb.LGBMRegressor(num_leaves = 31)
        
    elif type == 'binary':
        dici = prepare_classifier(dataf)
        estimator = lgb.LGBMClassifier(num_leaves = 31)
    else:
        dici = prepare_classifier(dataf)
        estimator = lgb.LGBMClassifier(num_leaves = 31)

    modl = estimator.fit(dici['X_train'], dici['y_train'],
                eval_set=[(dici['X_test'], dici['y_test'])],
                feature_name = dici['features'],
                early_stopping_rounds = 5)
    
    param_grid = {
        'learning_rate': np.linspace(0.01, 1, 7),
        'n_estimators': [150, 200, 250, 300],
        'boosting_type': ['gbdt', 'rf', 'dart'],
        'feature_fraction': [0.9],
        'bagging_fraction': [0.8],
        'bagging_freq': [5]
    }
        
    cv_modl = GridSearchCV(modl, param_grid, cv=3, verbose=5,n_jobs=6)
    cv_modl.fit(dici['X_train'], dici['y_train'])

    # Make directory if it doesn't exist yet
    Path(model_path / dep_var).mkdir(parents=True, exist_ok=True)
    modl.save_model(str(model_path / dep_var / "_extended.txt"))

    pickle.dump(dict['X_scaler'],
                open(model_path / dep_var / "_X_scaler_multi", 'wb'))




###############################################################################
if __name__ == "__main__":
    df = pd.read_pickle(input_path / 'merged').dropna()
    df1 = getdf(df)

    _estimate(df1, "employment_status", "multiclass")
    _estimate(df1, "hours", "regression")
    _estimate(df1, "gross_earnings", "regression")




