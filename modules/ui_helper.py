import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(chosen_set=None, set_file_dict=None):
    set_file_dict_item_lst = set_file_dict.items()
    for dataset, file_name in set_file_dict_item_lst:
        if chosen_set == dataset:
            df = pd.read_csv(file_name)
    return df


def make_slider(chosen_cl=None, cl_param_range_lst=None):
    param_val_lst = []
    for cl_param_range in cl_param_range_lst:
        cl, param, range_ = cl_param_range
        if chosen_cl == cl:
            val = st.sidebar.slider(param, *range_)
            param_val_lst.append((param, val))
    return param_val_lst


def extract_param_val_dict(param_val_lst=None):
    param_val_dict = {}
    for param_val_tpl in param_val_lst:
        param, val = param_val_tpl
        param_val_dict[param] = val
    return param_val_dict


def make_cl(chosen_cl=None, param_val_dict=None, name_cl_dict=None):
    name_cl_dict_item_lst = name_cl_dict.items()
    for cl_name, cl in name_cl_dict_item_lst:
        if chosen_cl == cl_name:
            cl = cl.set_params(**param_val_dict)
            return cl


def display_metric_data(name_metric_dict=None):
    item_lst = name_metric_dict.items()
    for name, metric in item_lst:
        st.write("\n", name, "->\n\n", metric)

 
def run_page(chosen_page=None, page_func_dict=None):
    page_func_dict_item_lst = page_func_dict.items()
    for page, func in page_func_dict_item_lst:
        if chosen_page == page:
            func()



