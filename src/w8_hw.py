import re
import sys
import category_encoders as ce
import numpy as np
import pandas as pd
from imblearn import over_sampling
from sklearn import (compose, ensemble, impute,
                     metrics, model_selection,
                     pipeline, preprocessing)


def main():
    file_name = "income_evaluation"
    file_ext = ".csv"
    file_name_ext = file_name + file_ext

    to_drop_col_name_arr = np.array(
        ["capital-gain", "capital-loss",
        "education", "fnlwgt", "native-country"]
        )

    df = pd.read_csv(file_name_ext)
    df = df.rename(str.strip, axis="columns")
    df = df.drop(
        labels=to_drop_col_name_arr, axis=1)
    df = df.replace(to_replace=re.compile(
        r" \?"), value=np.nan, regex=True)
    df = df.dropna()

    cont_col_name_arr = ["age", "hours-per-week"]
    binary_col_name_arr = ["workclass",
                   "marital-status", "occupation", "relationship", "race"]
    ordinal_col_name_arr = [
        "education-num", "sex", "income"]
    df_col_name_arr = df.columns

    log_transformer = preprocessing.FunctionTransformer(
        func=np.log10)
    imputer_pipe = pipeline.make_pipeline(impute.SimpleImputer(
        missing_values=r" \?", strategy="most_frequent"))
    cont_pipe = pipeline.make_pipeline(
        log_transformer, preprocessing.Normalizer())
    binary_pipe = pipeline.make_pipeline(
        ce.BinaryEncoder())
    ordinal_pipe = pipeline.make_pipeline(
        ce.OrdinalEncoder())
    col_transformer = compose.make_column_transformer(
        (cont_pipe,
         cont_col_name_arr), (binary_pipe, binary_col_name_arr),
        (ordinal_pipe, ordinal_col_name_arr)
    )
    over_sampler = over_sampling.SMOTE()
    
    col_transformer.fit(df)
    matrix = col_transformer.transform(df)

    X = matrix[:, :-1]
    y = matrix[:, -1]
    X, y = over_sampler.fit_resample(X, y)

    test_size = 0.2
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size)

    random_forest_cl = ensemble.RandomForestClassifier()
    random_forest_cl.fit(X_train, y_train)
    y_hat = random_forest_cl.predict(X_test)

    name_metric_dict = make_name_metric_dict(
        y=y_test, y_hat=y_hat)
    with open("w8_hw.txt", "w") as f:
        report_metric_data(
            name_metric_dict=name_metric_dict, file_obj=f)


def make_name_metric_dict(y=None, y_hat=None):
    accuracy = metrics.accuracy_score(
        y, y_hat)
    f1_score = metrics.f1_score(
        y, y_hat, zero_division=1)
    f2_score = metrics.fbeta_score(
        y, y_hat, beta=2, zero_division=1)
    conf_matrix = metrics.confusion_matrix(
        y, y_hat)
    cl_report = metrics.classification_report(
        y, y_hat, zero_division=1)
    roc_auc_score = metrics.roc_auc_score(
        y, y_hat)
    name_metric_dict = {"Accuracy score": accuracy, "F1 score": f1_score,
                        "F2 score": f2_score,
                        "Confusion matrix": conf_matrix,
                        "Classification report": cl_report,
                        "ROC-AUC score": roc_auc_score}
    return name_metric_dict


def report_metric_data(name_metric_dict=None, file_obj=None):
    item_lst = name_metric_dict.items()
    for name, metric in item_lst:
        print("\n", name, "->\n\n",
              metric, file=file_obj)


if __name__ == "__main__":
    main()
