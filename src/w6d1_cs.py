import catboost as catb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import ensemble, linear_model, naive_bayes, svm

from funcs import *


def main():
    f_space = pd.read_csv("data.csv")
    f_space = f_space.drop(
        labels=[" Net Income Flag", " Liability-Assets Flag"], axis=1)
    f_space_columns = f_space.columns
    for i in np.array(f_space.columns[1:]):
        if not (f_space[i] == 0).any():
            f_space = apply_transform(f_space, i)
    f_space = normalize(
        f_space, np.array(f_space.columns[1:]))
    voting_style = "hard"
    voting_ensemble = ensemble.VotingClassifier(estimators=[
        ("logistic_cl", linear_model.LogisticRegression(
            penalty="l1", solver="liblinear")),
        ("naive_bayes_cl", naive_bayes.GaussianNB()),
        ("random_forest_cl", ensemble.RandomForestClassifier())],
        voting=voting_style)
    bagging_ensemble_n_estimators = 10
    bagging_ensemble = ensemble.BaggingClassifier(base_estimator=svm.SVC(),
                                                  n_estimators=bagging_ensemble_n_estimators)
    gboosting_ensemble_n_estimators = 3
    gboosting_ensemble_learning_rate = 1.0
    gboosting_ensemble = ensemble.GradientBoostingClassifier(
        n_estimators=gboosting_ensemble_n_estimators,
        learning_rate=gboosting_ensemble_learning_rate)
    xgboost_ensemble_n_estimators = 3
    xgboost_ensemble = xgb.XGBClassifier(
        n_estimators=xgboost_ensemble_n_estimators,
        use_label_encoder=False, eval_metric="logloss")
    lightgboost_ensemble_n_estimators = 3
    lightgboost_ensemble = lgb.LGBMClassifier(
        n_estimators=lightgboost_ensemble_n_estimators)
    catboost_ensemble_iterations = 10
    catboost_ensemble_learning_rate = 1.0
    catboost_ensemble_depth = 1
    catboost_ensemble = catb.CatBoostClassifier(
        iterations=catboost_ensemble_iterations,
        learning_rate=catboost_ensemble_learning_rate,
        depth=catboost_ensemble_depth, verbose=False)
    ensemble_arr = np.array([linear_model.LogisticRegression(penalty="l1", solver="liblinear"),
                             voting_ensemble, bagging_ensemble,
                             gboosting_ensemble, xgboost_ensemble, lightgboost_ensemble,
                             catboost_ensemble])
    columns_without_first = f_space_columns[1:]
    predictor_arr = columns_without_first
    first_col_name = f_space_columns[0]
    predictor_arr = np.array(predictor_arr)
    outcome_feature = first_col_name
    for ensemble_algo in ensemble_arr:
        train_eval_model(ensemble_algo, f_space,
                         predictor_arr, outcome_feature)


if __name__ == "__main__":
    main()
