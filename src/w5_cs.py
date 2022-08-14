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

    csv_ = remove_column_name_ws(pd.read_csv("heart.csv"))

    csv_columns = csv_.columns

    f_space = csv_[["sex", "cp", "restecg", "exang", "ca", "thal", "target"]]
    
    f_space = delete_na(f_space, show_map=False)

    X = f_space.iloc[:, :-1].values

    Y = f_space.iloc[:, -1].values

    cl_names = ["SVM Linear Classifier", "SVM RBF Classifier", "Decision Tree Classifier", 
            "Random Forest Classifier", "Logistic Regression Classifier", "Naive Bayes Classifier", 
                "KNN Classifier"]

    svm_lin_cl = svm.SVC(kernel="linear")

    svm_rbf_cl = svm.SVC(kernel="rbf")

    decision_tree_cl = tree.DecisionTreeClassifier()

    random_forest_cl = ensemble.RandomForestClassifier()

    logistic_regression_cl = linear_model.LogisticRegression()

    naive_bayes_cl = naive_bayes.GaussianNB()

    k_neighbors_cl = neighbors.KNeighborsClassifier()   

    cl_arr = np.array([svm_lin_cl, svm_rbf_cl, decision_tree_cl,
            random_forest_cl, logistic_regression_cl,
                naive_bayes_cl, k_neighbors_cl])

    len_cl_names = len(cl_names)

    sum_f_1 = [0] * len_cl_names

    zero_conf_matr = [
                        [0, 0],
                        [0, 0]
                     ]

    sum_conf_matr = [zero_conf_matr] * len_cl_names

    n_iter = 20

    for j in range(n_iter):

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

        for i in cl_arr:

            i.fit(X_train, Y_train)
        
        f_arr = np.array([i.predict(X_test) for i in cl_arr])

        f_1_arr = np.array([metrics.f1_score(Y_test, i, average="micro") for i in f_arr])

        conf_mat_arr = np.array([metrics.confusion_matrix(Y_test, i) for i in f_arr])

        sum_f_1 += f_1_arr

        sum_conf_matr += conf_mat_arr

    for i in range(len_cl_names):

        print("\nAverage f1 score of", cl_names[i], " -> ", sum_f_1[i] / n_iter)

        print("\nIts confusion matrix -> \n", sum_conf_matr[i] / n_iter)
    
    min_ind = np.where(sum_f_1 == min(sum_f_1))[0][0]

    max_ind = np.where(sum_f_1 == max(sum_f_1))[0][0]

    print("\nAlgorithm with best f1 score ->", cl_names[max_ind], " ->", sum_f_1[max_ind] / n_iter)

    print("\nAlgorithm with worst f1 score ->", cl_names[min_ind], " ->", sum_f_1[min_ind] / n_iter)

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

