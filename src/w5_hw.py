import re
import tkinter

import category_encoders as ce
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling
import seaborn as sns
from imblearn import combine, over_sampling, under_sampling
from scipy import stats
from sklearn import (compose, ensemble, linear_model, metrics, model_selection,
                     naive_bayes, neighbors, preprocessing, svm, tree)


def main():

    matplotlib.use('TkAgg')

    ordinal_encoder = ce.OrdinalEncoder(cols=["income", "sex"])

    csv_ = remove_column_name_ws(pd.read_csv("income_evaluation.csv"))

    csv_ = csv_[["age", "sex", "education-num", "income"]]

    f_space = ordinal_encoder.fit_transform(delete_na(

        csv_.replace(to_replace=re.compile(r" \?"),
            
            value=np.nan, regex=True).replace(

                to_replace=re.compile(r"^ +| +$"), value="", regex=True),

                    show_map=False) )
    
    f_space["income"] -= 1

    f_space["sex"] -= 2

    f_space["sex"] = np.abs(f_space["sex"])

    sampler = over_sampling.SMOTE()

    X = f_space.iloc[:, :-1].values

    Y = f_space.iloc[:, -1].values

    X, Y = sampler.fit_resample(X, Y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

    classifier = ensemble.RandomForestClassifier()

    classifier.fit(X_train, Y_train)

    f = classifier.predict(X_test)

    f_1_score = metrics.f1_score(Y_test, f)

    print("confusion matrix -> \n", metrics.confusion_matrix(Y_test, f))

    print(metrics.classification_report(Y_test, f))

    print("ROC AUC score -> ", metrics.roc_auc_score(Y_test, f))

    plot_col_freq(f_space)


def delete_numeric_outliers(f_space):

    remove_indices = []

    for i in f_space:

        if not isinstance(f_space[i][0], str):

            bool_arr = np.abs(stats.zscore(f_space[i])) > 3

            for j in range(len(bool_arr)):

                if bool_arr[j]:

                    remove_indices.append(j)
    
    remove_indices = list(set(remove_indices))
    
    return f_space.drop(labels=remove_indices, axis=0)


def remove_column_name_ws(f_space):

    rename_dict = {}

    columns_arr = np.array(f_space.columns)

    for i in columns_arr:

        if i[0] is " ":

            rename_dict[i] = i.replace(" ", "")

    return f_space.rename(columns=rename_dict)


def delete_na(f_space, show_map=False):

    if show_map:

        sns.heatmap(f_space.isna())

        plt.show()

    return f_space.dropna()


def plot_col_freq(f_space):

    f_space_values = np.array(f_space.values)

    f_space_values_shape = np.array(f_space_values.shape)

    col_arr = np.reshape(f_space_values, f_space_values_shape)

    col_arr = col_arr.T

    last_col = col_arr[-1]

    col_arr = col_arr[:-1]

    col_names = f_space.columns     

    last_name = col_names[-1]
    
    plt.figure(num=last_name, figsize=[12.8, 12.8])

    sns.histplot(y=last_name, data=f_space)

    for index, elem in enumerate(col_arr):

        col_name = col_names[index]

        col = f_space[col_name]

        plt.figure(num=col_name, figsize=[12.8, 12.8])

        sns.histplot(y=col_name, hue=last_name,
            multiple="stack", data=f_space, kde=True)

    plt.show()


def write_xlsx(f_space):

    with open("income_evaluation.xlsx", "wb") as f:

        f_space.to_excel(f)


if __name__ == "__main__":

    main()



