import catboost as catb
import category_encoders as ce
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import ensemble, linear_model, naive_bayes, svm

from funcs import *


def main():
    file_name = "loan_prediction"
    file_extension = ".csv"
    full_file_name = file_name + file_extension
    f_space = pd.read_csv(full_file_name)
    raw_info(f_space)
    to_drop = np.array(["Loan_ID"])
    f_space = f_space.drop(labels=to_drop, axis=1)
    vis_missing(f_space)
    f_space = std_fillna(f_space)
    f_space = merge_columns(f_space, ("ApplicantIncome", "CoapplicantIncome"),
                            "TotalIncome", method=np.sum)
    to_apply_transform = np.array(
        ["TotalIncome", "LoanAmount"])
    f_space = apply_transform(
        f_space, to_apply_transform)
    to_normalize = np.array(
        ["TotalIncome", "LoanAmount", "Loan_Amount_Term"])
    f_space = normalize(f_space, to_normalize)
    corr_heatmap(f_space)
    plot_col_freq(f_space)
    f_space = encode_binary_col(f_space)
    encode_dict = {"Dependents": ce.OrdinalEncoder,
                   "Property_Area": ce.OneHotEncoder}
    f_space = encode_col(f_space, encode_dict)
    f_space["Dependents_0"] -= 1
    write_xlsx(f_space, file_name)
    f_space_columns = f_space.columns
    voting_style = "hard"
    voting_ensemble = ensemble.VotingClassifier(estimators=[
        ("logistic_cl", linear_model.LogisticRegression(
            penalty="l1", solver="liblinear")),
        ("naive_bayes_cl", naive_bayes.GaussianNB()),
        ("random_forest_cl", ensemble.RandomForestClassifier())],
        voting=voting_style)
    bagging_ensemble_n_estimators = 100
    bagging_ensemble = ensemble.BaggingClassifier(base_estimator=svm.SVC(),
                                                  n_estimators=bagging_ensemble_n_estimators)
    gboosting_ensemble_n_estimators = 6
    gboosting_ensemble_learning_rate = 1.0
    gboosting_ensemble = ensemble.GradientBoostingClassifier(
        n_estimators=gboosting_ensemble_n_estimators,
        learning_rate=gboosting_ensemble_learning_rate)
    xgboost_ensemble_n_estimators = 6
    xgboost_ensemble = xgb.XGBClassifier(
        n_estimators=xgboost_ensemble_n_estimators,
        use_label_encoder=False, eval_metric="logloss")
    lightgboost_ensemble_n_estimators = 6
    lightgboost_ensemble = lgb.LGBMClassifier(
        n_estimators=lightgboost_ensemble_n_estimators)
    catboost_ensemble_iterations = 100
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
    columns_without_last = f_space_columns[:-1]
    predictor_arr = columns_without_last
    last_col_name = f_space_columns[-1]
    predictor_arr = np.array(predictor_arr)
    outcome_feature = last_col_name
    for ensemble_algo in ensemble_arr:
        train_eval_model(ensemble_algo, f_space,
                         predictor_arr, outcome_feature)
    plt.show()


if __name__ == "__main__":
    main()
