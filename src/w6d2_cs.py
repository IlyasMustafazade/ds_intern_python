import warnings as warnings

import category_encoders as ce
import numpy as np
import pandas as pd
import skopt as skopt
import xgboost as xgb

from funcs import *

warnings.filterwarnings(
    "ignore", category=UserWarning)


def main():
    file_name = "nasa"
    file_extension = ".csv"
    full_file_name = file_name + file_extension
    f_space = pd.read_csv(full_file_name)
    f_space["Hazardous"] = f_space["Hazardous"].astype(
        int)
    f_space = f_space.select_dtypes(
        include=np.number)
    f_space_columns = f_space.columns
    f_space_columns_without_last = f_space_columns[:-1]
    last_column_name = f_space_columns[-1]
    f_space_columns_without_last_2 = f_space_columns[:-2]
    f_space_columns_without_last_as_tuple = tuple(
        f_space_columns_without_last)
    predictor_vector = f_space[[
        *f_space_columns_without_last_as_tuple]]
    outcome_feature = f_space[last_column_name]
    alpha_val_random = 10
    xgboost_cl_for_random = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="error", alpha=alpha_val_random)
    alpha_val_bayesian = 100000
    xgboost_cl_for_bayesian = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="error", alpha=alpha_val_bayesian)
    xgboost_cl_param_dict = {
        "booster": ["gbtree", "gblinear", "dart"],
        "n_estimators": np.arange(5, 100, 20),
        "max_depth": [None]
    }
    best_hparams_bayesian = get_best_hparams(predictor_vector=predictor_vector, outcome_feature=outcome_feature,
                                             algo=xgboost_cl_for_bayesian, param_dict=xgboost_cl_param_dict,
                                             tuner_class="bayesian", cv=2, n_iter=3,
                                             verbose=True)
    best_hparams_randomized = get_best_hparams(predictor_vector=predictor_vector, outcome_feature=outcome_feature,
                                               algo=xgboost_cl_for_random, param_dict=xgboost_cl_param_dict,
                                               tuner_class="randomized", cv=5, n_iter=5,
                                               verbose=True)
    random_tuned_xgboost = xgboost_cl_for_random
    random_tuned_xgboost.set_params(
        **best_hparams_randomized)
    bayesian_tuned_xgboost = xgboost_cl_for_bayesian
    bayesian_tuned_xgboost.set_params(
        **best_hparams_bayesian)
    algo_arr = np.array(
        [random_tuned_xgboost, bayesian_tuned_xgboost])
    for algo in algo_arr:
        train_eval_model(algo, f_space,
                         f_space_columns_without_last,
                         last_column_name, verbose=True)


if __name__ == "__main__":
    main()
