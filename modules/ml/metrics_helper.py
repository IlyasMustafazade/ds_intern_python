from sklearn import metrics


def make_name_metric_dict(y=None, y_hat=None):
    accuracy = metrics.accuracy_score(y, y_hat)
    f1_score = metrics.f1_score(y, y_hat, zero_division=1)
    f2_score = metrics.fbeta_score(y, y_hat, beta=2, zero_division=1)
    conf_matrix = metrics.confusion_matrix(y, y_hat)
    cl_report = metrics.classification_report(y, y_hat, zero_division=1)
    roc_auc_score = metrics.roc_auc_score(y, y_hat)
    name_metric_dict = {"Accuracy score": accuracy, "F1 score": f1_score,
        "F2 score": f2_score, "Confusion matrix": conf_matrix,
        "Classification report": cl_report, "ROC-AUC score": roc_auc_score}
    return name_metric_dict


def report_metric_data(name_metric_dict=None, file_obj=None):
    item_lst = name_metric_dict.items()
    for name, metric in item_lst:
        print("\n", name, "->\n\n", metric, file=file_obj)
