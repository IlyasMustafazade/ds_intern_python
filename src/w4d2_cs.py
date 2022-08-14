import re
from datetime import datetime

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import (compose, linear_model, metrics, model_selection,
                     preprocessing)

def main():

    cond_arr = ("Confirmed", "Inflected", "Recovered", "Deaths") 

    f_space = pd.read_csv("covid_19_data.csv").dropna()

    f_space["Inflected"] = f_space["Confirmed"] - f_space["Recovered"] - f_space["Deaths"]

    f_space["ObservationDate"] = pd.to_datetime(f_space['ObservationDate'], format="%m/%d/%Y")

    f_space["ObservationDate"] = (f_space.ObservationDate - f_space.ObservationDate.min()).astype("timedelta64[D]")

    f_space = f_space.drop(columns=["Last Update", "SNo"])

    transformer = ce.BinaryEncoder(cols=["Province/State", "Country/Region"])

    transformer.fit(f_space)

    f_space = pd.DataFrame(transformer.transform(f_space))

    with open("covid_19_data.xlsx", "wb") as f: f_space.to_excel(f)

    grouped = f_space.groupby("ObservationDate")

    new_f_space = pd.DataFrame(data={"ObservationDate": [], "Confirmed": [],
                        "Inflected": [], "Recovered": [], "Deaths": []})

    for name, group in grouped:

        new_f_space.loc[len(new_f_space.index)] = [name, group["Confirmed"].sum(),
            group["Inflected"].sum(), group["Recovered"].sum(),
                group["Deaths"].sum()] 
    
    f_space = new_f_space

    X = f_space[["ObservationDate"]]

    Y_confirmed, Y_inflected, Y_recovered, Y_death = f_space[["Confirmed"]], \
        f_space[["Inflected"]], f_space[["Recovered"]], f_space[["Deaths"]]

    Y_arr = np.array((Y_confirmed, Y_inflected, Y_recovered, Y_death))

    train_test_arr = []

    for i in Y_arr:

        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, i, test_size=0.2)

        train_test_arr.append([x_train.to_numpy(), x_test.to_numpy(), y_train, y_test])
    
    regressor_arr = [linear_model.LinearRegression() for i in range(len(Y_arr))]

    for i in range(len(regressor_arr)):

        regressor_arr[i].fit(train_test_arr[i][0], train_test_arr[i][2])

    estimator_arr = []

    for i in range(len(regressor_arr)):

        estimator_arr.append(regressor_arr[i].coef_ *
            np.arange(70) + regressor_arr[i].intercept_)

    plt_arr = []

    for i in range(len(train_test_arr)):

        plt_arr.append(plt.scatter(X, Y_arr[i]))
    
    for i in estimator_arr:

        plt_arr.append(plt.plot(np.arange(70), i.flatten()))

    plt.legend(plt_arr, cond_arr)

    result_arr = []

    for i in estimator_arr:

        result_arr.append(i.flatten()[63])
    
    estimation_plt = plt.scatter([63] * 4, result_arr, color="black")

    for i in range(len(result_arr)):
        print("Predicted", cond_arr[i], "->", result_arr[i])

    plt.show()


if __name__ == "__main__": main()





