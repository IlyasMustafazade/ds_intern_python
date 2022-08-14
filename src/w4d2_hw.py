import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import (compose, linear_model, metrics, model_selection,
                     preprocessing)


def main():

    f_space = pd.read_csv("USA_Housing.csv")

    transformer = ce.BaseNEncoder(cols=["Address"])

    transformer.fit(f_space)

    encoded_f_space = pd.DataFrame(transformer.transform(f_space))

    X = encoded_f_space.drop(columns="Price")
    
    Y = encoded_f_space[["Price"]]

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

    regressor = linear_model.LinearRegression()

    regressor.fit(X_train, Y_train)

    f = regressor.predict(X_test)

    r_2 = metrics.r2_score(Y_test, f)

    print("r squared score -> ", r_2 * 100, " %")

    print("adjusted r squared score -> ", adjusted_r_2(r_2, encoded_f_space) * 100, " %")

    print("mean absolute error -> ", metrics.mean_absolute_error(Y_test, f))
    
    Y_test = Y_test.values.flatten()

    f = f.flatten()

    pred_act_df = pd.DataFrame(data={"Actual": Y_test, "Predicted": f})

    print(pred_act_df.head())

    intercept, coef = regressor.intercept_.flatten(), regressor.coef_.flatten()

    print("intercept -> ", intercept)

    print("coef -> ", coef)

    # Enter features to get estimation:

    features = np.reshape(np.array((65000, 6, 7, 4, 47000, 
                "37778 George Ridges Apt. 509\nEast Holly, NV 29290-3595")), (-1, 1))

    try: 

        ind = list(f_space["Address"].values).index(features[-1][0])
    
    except ValueError:

        print("Please enter valid address")

    features = features[:-1]

    features = np.append(features, np.reshape(encoded_f_space.iloc[ind, 6:].values, (1, -1)).flatten())

    features = features.astype("float64")

    print("Estimate -> ", intercept + features @ coef)
    
    pred_act_df.head(20).plot(kind="bar")

    plt.show()


def adjusted_r_2(r_2, f_space):

    n, p = f_space.shape[0], f_space.shape[1]

    return 1 - (1 - r_2) / (n - p) * (n - 1)


def clean_outlier(f_space):

    copy_f_space = f_space

    return copy_f_space[(np.abs(stats.zscore(copy_f_space)) < 3).all(axis=1)]


if __name__=="__main__":

    main()


