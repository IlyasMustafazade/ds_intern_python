import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import (StandardScaler,
             RobustScaler, MinMaxScaler, LabelEncoder)
from sklearn import metrics
from PIL import Image
from category_encoders import BinaryEncoder, OneHotEncoder
from pandas.core.common import SettingWithCopyWarning
from pandas.api.types import is_numeric_dtype  
from scipy.stats import zscore

def main():
    dsa_logo = Image.open("dsa_logo.png")
    web_app = WebApp()

    st.sidebar.image(dsa_logo)
    web_app.chosen_page = st.sidebar.selectbox("", web_app.PAGE_TPL)

    run_page(chosen_page=web_app.chosen_page, page_func_dict=web_app.PAGE_FUNC_DICT)

class WebApp():
    def __init__(self):
        self.PAGE_TPL = ("Homepage", "EDA", "Modeling")
        self.FUNC_TPL = (self.run_home_page, self.run_eda_page, self.run_modeling_page)
        len_page_tpl = len(self.PAGE_TPL)
        self.PAGE_FUNC_DICT = {self.PAGE_TPL[i]: self.FUNC_TPL[i] for i in range(len_page_tpl)}
        self.SET_TPL = ("Water potability", "Loan prediction")
        self.FILE_TPL = ("water_potability.csv", "loan_prediction.csv")
        self.DF_TPL = [pd.read_csv(f) for f in self.FILE_TPL]
        self.DF_TPL[1] = adjust_loan_set(df=self.DF_TPL[1])
        self.DF_TPL = tuple(self.DF_TPL)
        len_set_tpl = len(self.SET_TPL)
        self.SET_DF_DICT = {
            self.SET_TPL[i]: self.DF_TPL[i] for i in range(len_set_tpl)}
        self.SCALE_MODE_TPL = ("Standard", "Robust", "MinMax")
        self.ENCODE_MODE_TPL = ("Binary", "One-Hot")
        self.CL_NAME_TPL = ("XGBoost", "Naive Bayes",
                            "Logistic Regression", "Random Forest",
                            "Support Vector")
        self.CL_TPL = (XGBClassifier, GaussianNB, LogisticRegression,
                       RandomForestClassifier, SVC)
        len_model_tpl = len(self.CL_TPL)
        self.NAME_CL_DICT = {self.CL_NAME_TPL[i]: self.CL_TPL[i] for i in range(len_model_tpl)}
        self.CAT_IMPUTE_MODE_TPL = ("Mode", "Backfill", "Forwardfill")
        self.NUM_IMPUTE_MODE_TPL = ("Mode", "Median")
        self.FEATURE_ENG_MODE_TPL = ("Under Sampling", "Clean Outlier")

        self.chosen_page = self.run_home_page
        self.chosen_set = self.SET_TPL[0]
        self.chosen_cat_impute_mode = self.CAT_IMPUTE_MODE_TPL[0]
        self.chosen_num_impute_mode = self.NUM_IMPUTE_MODE_TPL[0]
        self.chosen_feature_eng_mode = self.FEATURE_ENG_MODE_TPL[0]
        self.chosen_scale_mode = self.SCALE_MODE_TPL[0]
        self.chosen_encode_mode = self.ENCODE_MODE_TPL[0]
        self.chosen_random_state = 0
        self.chosen_test_size = 20
        self.chosen_cl = self.CL_NAME_TPL[0] 

        self.SAVED_CHOICE_TPL = ("chosen_cat_impute_mode",
            "chosen_num_impute_mode", "chosen_feature_eng_mode",
            "chosen_scale_mode", "chosen_encode_mode",
            "random_state", "test_size", "chosen_cl")
        self.CHOICE_LST = [self.chosen_cat_impute_mode, 
            self.chosen_num_impute_mode, self.chosen_feature_eng_mode, 
            self.chosen_scale_mode, self.chosen_encode_mode, 
            self.chosen_random_state, self.chosen_test_size, self.chosen_cl]
        self.N_CHOICE = len(self.CHOICE_LST)

        self.df = None

    def run_home_page(self):
        self.chosen_set = self.make_home_ui()
        st.session_state.chosen_set = self.chosen_set

        self.df = self.make_df()

    def run_eda_page(self):
        self.chosen_set = st.session_state.chosen_set
        self.chosen_cat_impute_mode, self.chosen_num_impute_mode, \
          self.chosen_feature_eng_mode = self.make_eda_ui()
        self.save_choice_lst()

    def run_modeling_page(self):
        self.chosen_set = st.session_state.chosen_set
        self.make_modeling_ui()

    def run_model_button(self):
        y_test, y_hat = self.train_predict()
        name_metric_dict = make_name_metric_dict(y=y_test, y_hat=y_hat)
        display_metric_data(name_metric_dict=name_metric_dict)
    
    def train_predict(self):
        cl_class = self.NAME_CL_DICT[self.chosen_cl]
        if self.chosen_cl == "XGBoost":
            cl = cl_class(use_label_encoder=False, eval_metric="logloss")
        else:
            cl = cl_class()
        X, y = extract_X_y_df(df=self.df)
        fraction_test_size = float(self.chosen_test_size) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y,
            random_state=int(self.chosen_random_state), test_size=fraction_test_size)
        
        cl.fit(X_train, y_train)
        y_hat = cl.predict(X_test)
        return y_test, y_hat

    def make_home_ui(self):
        ds_role = Image.open("ds_role.png")

        st.title("Homepage")
        st.image(ds_role)
        chosen_set = st.selectbox("Select dataset", self.SET_TPL)
        st.write("Selected: **" + chosen_set + "** dataset") 

        return chosen_set

    def make_eda_ui(self):
        self.fetch_choice_lst()
        self.df = self.make_df()
        original_df = self.SET_DF_DICT[self.chosen_set]
        y = extract_X_y_df(df=original_df)[1]
        na_original_df = original_df.isna()
        na_count_arr = na_original_df.sum()
        na_count = na_count_arr.sum()
        m, n = original_df.shape
        percent_na = ((na_count / m) / n) * 100
        percent_na = round(percent_na, 3)
        percent_na = str(percent_na)

        st.title("EDA")
        st.dataframe(self.df.head(100))
        st.markdown(""" 
        # Statistical Values
        """)
        st.dataframe(self.df.describe())

        fig = plt.figure()
        sns.histplot(y)
        st.markdown(""" 
        # Balance of Data
        """)
        st.pyplot(fig=fig)

        st.markdown(""" 
        # Missing values heatmap
        """)
        fig = plt.figure()
        sns.heatmap(na_original_df)
        st.pyplot(fig=fig)

        st.markdown(""" 
        # Percentage of missing values 
        """)
        st.write("**" + percent_na + "** %")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(""" 
            # Null Variables
            """)     
            st.write(na_count_arr)
        with col2:
            st.markdown(""" 
            # Imputation
            """)
            chosen_cat_impute_mode = st.radio("Categorical", self.CAT_IMPUTE_MODE_TPL)

            chosen_num_impute_mode = st.radio("Numerical", self.NUM_IMPUTE_MODE_TPL)

            st.markdown(""" 
            # Feature Engineering
            """)
            chosen_feature_eng_mode = st.radio("", self.FEATURE_ENG_MODE_TPL)

            st.button("Data preprocessing", on_click=self.make_df)
        choice_tpl = (chosen_cat_impute_mode, chosen_num_impute_mode, chosen_feature_eng_mode)
        return choice_tpl

    def make_modeling_ui(self):
        self.fetch_choice_lst()
        self.df = self.make_df()

        st.title("Modeling")
        st.dataframe(self.df.head(10))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(""" 
            # Scaling
            """)
            chosen_scale_mode = st.radio("", self.SCALE_MODE_TPL)
        
        with col2:
            st.markdown(""" 
            # Encoders
            """)
            chosen_encode_mode = st.radio("", self.ENCODE_MODE_TPL)
        
        st.markdown(""" 
            # Train and Test Splitting
            """)

        col1, col2 = st.columns(2)
        with col1:
            chosen_random_state = st.text_input("Random State")
        with col2:
            chosen_test_size = st.text_input("Percentage")
        
        chosen_cl = st.selectbox("Select Model", self.CL_NAME_TPL)
        st.write("Selected: **" + chosen_cl + "** Model")
        choice_tpl = (chosen_scale_mode, chosen_encode_mode,
                      chosen_random_state, chosen_test_size, chosen_cl)
        self.chosen_scale_mode, self.chosen_encode_mode, \
                      self.chosen_random_state, self.chosen_test_size, \
                        self.chosen_cl = choice_tpl
        self.save_choice_lst()
        self.df = self.make_df()
        st.write(st.session_state)
        is_clicked = st.button("Run Model")
        if is_clicked is True:
            self.run_model_button()

    def make_df(self):
        df = self.SET_DF_DICT[self.chosen_set]
        pipe = (self.apply_feature_eng_mode, self.apply_num_impute_mode,
                self.apply_cat_impute_mode, self.apply_encode_mode, 
                self.apply_scale_mode)
        previous_df = df
        current_df = None
        for process in pipe:
            current_df = process(df=previous_df)
            previous_df = current_df.copy()
        if current_df.isna().sum().sum():
            raise Exception(
                "Imputed DataFrame object can not include NaN"
            )
        return current_df

    def apply_feature_eng_mode(self, df=None):
        if self.chosen_feature_eng_mode == "Under Sampling":
            X, y = extract_X_y_df(df=df)
            sampler = RandomUnderSampler(random_state=0)
            X, y = sampler.fit_resample(X, y)
            new_df = pd.concat([X, y], axis=1)
            return new_df
        elif self.chosen_feature_eng_mode == "Clean Outlier":
            new_df = del_num_outlier(df=df)
            return new_df

    def apply_num_impute_mode(self, df=None):
        num_df = df.select_dtypes(include="number")
        num_col_idx = num_df.columns
        num_col_lst = list(num_col_idx.to_numpy())
        if len(num_col_lst) < 1:
            return df
        if self.chosen_num_impute_mode == "Mode":
            imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        elif self.chosen_num_impute_mode == "Median":
            imputer = SimpleImputer(missing_values=np.nan, strategy="median")

        imputer.fit(num_df)
        matrix = imputer.transform(num_df)
        imputed_df = pd.DataFrame(data=matrix, columns=num_col_lst)
        new_df = df.copy()
        new_df.update(imputed_df)
        return new_df

    def apply_cat_impute_mode(self, df=None):
        cat_df = df.select_dtypes(include=object)
        cat_col_idx = cat_df.columns
        cat_col_lst = list(cat_col_idx.to_numpy())
        if len(cat_col_lst) < 1:
            return df
        if self.chosen_cat_impute_mode == "Mode":
            imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            imputer.fit(cat_df)
            matrix = imputer.transform(cat_df)
            imputed_df = pd.DataFrame(data=matrix, columns=cat_col_lst)
            new_df = df
            new_df.update(imputed_df)
        elif self.chosen_cat_impute_mode == "Backfill":
            new_df = df.bfill(axis=1)
        elif self.chosen_cat_impute_mode == "Forwardfill":
            new_df = df.ffill(axis=1)
        return new_df

    def apply_encode_mode(self, df=None):
        name_encoder_lst = [("Binary", BinaryEncoder()),
                            ("One-Hot", OneHotEncoder())]
        for name_encoder in name_encoder_lst:
            name, encoder = name_encoder
            if self.chosen_scale_mode == name:
                chosen_encoder = encoder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoder.fit(df)
            matrix = encoder.transform(df)
        column_arr = encoder.get_feature_names()
        new_df = pd.DataFrame(data=matrix, columns=column_arr)
        return new_df

    def apply_scale_mode(self, df=None):
        name_scaler_lst = [("Standard", StandardScaler()),
                           ("Robust", RobustScaler()),
                           ("MinMax", MinMaxScaler())]
        for name_scaler in name_scaler_lst:
            name, scaler = name_scaler
            if self.chosen_scale_mode == name:
                chosen_scaler = scaler
        df_last_name = df.columns[-1]
        X = df.iloc[:, :-1]
        y = df[[df_last_name]]
        y = y.astype(dtype=np.int8, errors="ignore")
        scaler.fit(X)
        matrix = scaler.transform(X)
        column_arr = scaler.get_feature_names_out()
        X = pd.DataFrame(data=matrix, columns=column_arr)
        new_df = pd.concat([X, y], axis=1)
        return new_df

    def fetch_choice_lst(self):
        can_load = True
        for choice in self.SAVED_CHOICE_TPL:
            if not (choice in st.session_state):
                can_load = False
        if can_load is False:
            return
        self.chosen_cat_impute_mode = st.session_state.chosen_cat_impute_mode
        self.chosen_num_impute_mode = st.session_state.chosen_num_impute_mode
        self.chosen_feature_eng_mode = st.session_state.chosen_feature_eng_mode
        self.chosen_scale_mode = st.session_state.chosen_scale_mode
        self.chosen_encode_mode = st.session_state.chosen_encode_mode
        self.chosen_random_state = st.session_state.random_state
        self.chosen_test_size = st.session_state.test_size
        self.chosen_cl = st.session_state.chosen_cl 

    def save_choice_lst(self):
        st.session_state.chosen_cat_impute_mode = self.chosen_cat_impute_mode
        st.session_state.chosen_num_impute_mode = self.chosen_num_impute_mode
        st.session_state.chosen_feature_eng_mode = self.chosen_feature_eng_mode
        st.session_state.chosen_scale_mode = self.chosen_scale_mode
        st.session_state.chosen_encode_mode = self.chosen_encode_mode
        st.session_state.random_state = self.chosen_random_state
        st.session_state.test_size = self.chosen_test_size
        st.session_state.chosen_cl = self.chosen_cl

    def display_choice_lst(self):
        st.write(self.chosen_cat_impute_mode)
        st.write(self.chosen_num_impute_mode)
        st.write(self.chosen_feature_eng_mode)
        st.write(self.chosen_scale_mode)
        st.write(self.chosen_encode_mode)
        st.write(self.chosen_random_state)
        st.write(self.chosen_test_size)
        st.write(self.chosen_cl)

def adjust_loan_set(df=None):
    copy_df = df
    column_arr = df.columns
    last_name = column_arr[-1]
    last_column = df[last_name]
    label_encoder = LabelEncoder()
    vector = label_encoder.fit_transform(last_column)
    series = pd.Series(data=vector, name=last_name, dtype=np.int8)
    copy_df.loc[:, last_name] = series
    return copy_df

def del_num_outlier(df=None, col_arr=None, stddev_limit=3):
    remove_indices = []
    for column_name in df:
        col_as_arr = df[column_name]
        col_as_arr = np.array(col_as_arr)
        if is_numeric_dtype(col_as_arr):
            bool_arr = np.abs(zscore(
                df[column_name])) > stddev_limit
            bool_arr = np.array(bool_arr)
            for j in range(len(bool_arr)):
                if bool_arr[j] is True:
                    remove_indices.append(j)
    remove_indices = list(set(remove_indices))
    remove_indices = np.array(remove_indices, dtype=int)
    return df.drop(labels=df.index[remove_indices], axis=0)

def run_page(chosen_page=None, page_func_dict=None):
    page_func_dict_item_lst = page_func_dict.items()
    for page, func in page_func_dict_item_lst:
        if chosen_page == page:
            func()

def display_metric_data(name_metric_dict=None):
    item_lst = name_metric_dict.items()
    for name, metric in item_lst:
        st.write("\n", name, "->\n\n", metric)

def extract_X_y_df(df=None):
    column_name_arr = df.columns
    column_name_arr_without_last = df.columns[:-1]
    column_name_tpl_without_last = tuple(column_name_arr_without_last)
    last_column_name = column_name_arr[-1]
    X = df[[*column_name_tpl_without_last]]
    y = df[last_column_name]
    return X, y

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


if __name__ == "__main__":
    main()
