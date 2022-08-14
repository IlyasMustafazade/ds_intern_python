import numpy as np
import pandas as pd
import skopt as skopt
import xgboost as xgb
from skopt import space

from funcs import *


def main():
    file_name = "diabetes"
    file_extension = ".csv"
    full_file_name = file_name + file_extension
    f_space = pd.read_csv(full_file_name)
    raw_info(f_space)
    vis_missing(f_space)
    corr_heatmap(f_space)
    plot_col_freq(f_space)
    f_space_columns = f_space.columns
    f_space_columns_without_last = f_space_columns[:-1]
    last_column_name = f_space_columns[-1]
    f_space_columns_without_last_2 = f_space_columns[:-2]
    to_delete_zero = np.array(
        ["BloodPressure", "SkinThickness", "BMI"])
    f_space = remove_rows_with_val(
        f_space=f_space, col_arr=to_delete_zero, val=0)
    to_small_increment = np.array(
        f_space_columns_without_last)
    f_space = apply_transform(
        f_space, to_small_increment, method=small_increment)
    to_log_transform = np.array(
        f_space_columns_without_last_2)
    f_space = apply_transform(
        f_space, to_log_transform, method=np.log10)
    to_normalize = np.array(
        [f_space_columns_without_last_2])
    f_space = normalize(f_space, to_normalize)
    to_delete_outliers = f_space_columns_without_last
    f_space = delete_numeric_outliers(
        f_space, to_delete_outliers, stddev_limit=4)
    write_xlsx(f_space, file_name)
    f_space_columns_without_last_as_tuple = tuple(
        f_space_columns_without_last)
    predictor_vector = f_space[[
        *f_space_columns_without_last_as_tuple]]
    outcome_feature = f_space[last_column_name]
    xgboost_cl = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="error")
    xgboost_cl_param_dict = {
        "booster": ["gbtree", "gblinear", "dart"],
        "n_estimators": np.arange(5, 100, 5),
    }
    best_hparams_grid = get_best_hparams(predictor_vector=predictor_vector, outcome_feature=outcome_feature,
                                         algo=xgboost_cl,
                                         param_dict=xgboost_cl_param_dict,
                                         tuner_class="grid", cv=10, n_iter=10, verbose=True)
    best_hparams_random = get_best_hparams(predictor_vector=predictor_vector, outcome_feature=outcome_feature,
                                           algo=xgboost_cl, param_dict=xgboost_cl_param_dict,
                                           tuner_class="randomized", cv=10, n_iter=10,
                                           verbose=True)
    grid_tuned_xgboost = xgboost_cl
    grid_tuned_xgboost.set_params(
        **best_hparams_grid)
    random_tuned_xgboost = xgboost_cl
    random_tuned_xgboost.set_params(
        **best_hparams_random)
    algo_arr = np.array(
        [grid_tuned_xgboost, random_tuned_xgboost])
    for algo in algo_arr:
        train_eval_model(algo, f_space,
                         f_space_columns_without_last,
                         last_column_name, verbose=True)
    plt.show()


if __name__ == "__main__":
    main()
