import os
import sys
from tqdm import tqdm

import numpy as np
import scipy
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness

from warnings import simplefilter
simplefilter(action='ignore')

g = Group()
b = Bias()

icu_dataset = pd.read_csv("../WiDS2021/wids2021_dataset_cleaned_v3.csv")
protected_attributes = pd.read_csv("../WiDS2021/protected_attributes.csv")["ethnicity"]

X = icu_dataset.drop("diabetes_mellitus", axis=1)
y = icu_dataset["diabetes_mellitus"].rename("label_value")

n_cpu = os.cpu_count()

lr_space = dict(C=scipy.stats.uniform(0.1, 10))

metrics = {
    "accuracy": [],
    "roc_auc": [],
    "f1": [],
    "fpr_Caucasian": [],
    "fprd_Caucasian": [],
    "fpr_AfricanAmerican": [],
    "fprd_AfricanAmerican": [],
    "fpr_Asian": [],
    "fprd_Asian": [],
    "fpr_Hispanic": [],
    "fprd_Hispanic": [],
    "fpr_NativeAmerican": [],
    "fprd_NativeAmerican": [],
    "fpr_Other": [],
    "fprd_Other": [],
    "fnr_Caucasian": [],
    "fnrd_Caucasian": [],
    "fnr_AfricanAmerican": [],
    "fnrd_AfricanAmerican": [],
    "fnr_Asian": [],
    "fnrd_Asian": [],
    "fnr_Hispanic": [],
    "fnrd_Hispanic": [],
    "fnr_NativeAmerican": [],
    "fnrd_NativeAmerican": [],
    "fnr_Other": [],
    "fnrd_Other": [],
}

for experiment_seed in tqdm(range(10)):

    np.random.seed(experiment_seed)

    cv_outer = StratifiedKFold(
        n_splits=3,
        shuffle=True,
    )

    yhats = pd.Series(dtype="float64")

    for train_idx, test_idx in tqdm(cv_outer.split(X, y), leave=False):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
        # CV for hyperparameter-tunning
        cv_inner = StratifiedKFold(n_splits=2, shuffle=True)
        model = LogisticRegression(class_weight="balanced")
        search = RandomizedSearchCV(model, lr_space, scoring='f1', cv=cv_inner, refit=True, n_iter=10)
        result = search.fit(X_train, y_train)
    
        # Best found model
        best_model = result.best_estimator_
        yhat = best_model.predict(X_test)
        yhat_index = pd.Series(yhat, index=y_test.index).rename("score")
        yhats = pd.concat([yhats, yhat_index])

    yhats = yhats.sort_index().rename("score")

    fairness_df = pd.concat([y, yhats, protected_attributes], axis=1, join="inner")
    xtab, _ = g.get_crosstabs(fairness_df, attr_cols=["ethnicity"])
    bdf = b.get_disparity_predefined_groups(
        xtab,
        original_df=fairness_df,
        ref_groups_dict={'ethnicity':'Caucasian'},
        alpha=0.05,
        check_significance=True, 
        mask_significance=True
    )


    metrics["accuracy"].append(accuracy_score(y, yhats))
    metrics["roc_auc"].append(roc_auc_score(y, yhats))
    metrics["f1"].append(f1_score(y, yhats))
    metrics["fpr_Caucasian"].append(bdf.loc[bdf["attribute_value"]=="Caucasian"]["fpr"].values[0])
    metrics["fprd_Caucasian"].append(bdf.loc[bdf["attribute_value"]=="Caucasian"]["fpr_disparity"].values[0])
    metrics["fpr_AfricanAmerican"].append(bdf.loc[bdf["attribute_value"]=="African American"]["fpr"].values[0])
    metrics["fprd_AfricanAmerican"].append(bdf.loc[bdf["attribute_value"]=="African American"]["fpr_disparity"].values[0])
    metrics["fpr_Asian"].append(bdf.loc[bdf["attribute_value"]=="Asian"]["fpr"].values[0])
    metrics["fprd_Asian"].append(bdf.loc[bdf["attribute_value"]=="Asian"]["fpr_disparity"].values[0])
    metrics["fpr_Hispanic"].append(bdf.loc[bdf["attribute_value"]=="Hispanic"]["fpr"].values[0])
    metrics["fprd_Hispanic"].append(bdf.loc[bdf["attribute_value"]=="Hispanic"]["fpr_disparity"].values[0])
    metrics["fpr_NativeAmerican"].append(bdf.loc[bdf["attribute_value"]=="Native American"]["fpr"].values[0])
    metrics["fprd_NativeAmerican"].append(bdf.loc[bdf["attribute_value"]=="Native American"]["fpr_disparity"].values[0])
    metrics["fpr_Other"].append(bdf.loc[bdf["attribute_value"]=="Other/Unknown"]["fpr"].values[0])
    metrics["fprd_Other"].append(bdf.loc[bdf["attribute_value"]=="Other/Unknown"]["fpr_disparity"].values[0])
    metrics["fnr_Caucasian"].append(bdf.loc[bdf["attribute_value"]=="Caucasian"]["fnr"].values[0])
    metrics["fnrd_Caucasian"].append(bdf.loc[bdf["attribute_value"]=="Caucasian"]["fnr_disparity"].values[0])
    metrics["fnr_AfricanAmerican"].append(bdf.loc[bdf["attribute_value"]=="African American"]["fnr"].values[0])
    metrics["fnrd_AfricanAmerican"].append(bdf.loc[bdf["attribute_value"]=="African American"]["fnr_disparity"].values[0])
    metrics["fnr_Asian"].append(bdf.loc[bdf["attribute_value"]=="Asian"]["fnr"].values[0])
    metrics["fnrd_Asian"].append(bdf.loc[bdf["attribute_value"]=="Asian"]["fnr_disparity"].values[0])
    metrics["fnr_Hispanic"].append(bdf.loc[bdf["attribute_value"]=="Hispanic"]["fnr"].values[0])
    metrics["fnrd_Hispanic"].append(bdf.loc[bdf["attribute_value"]=="Hispanic"]["fnr_disparity"].values[0])
    metrics["fnr_NativeAmerican"].append(bdf.loc[bdf["attribute_value"]=="Native American"]["fnr"].values[0])
    metrics["fnrd_NativeAmerican"].append(bdf.loc[bdf["attribute_value"]=="Native American"]["fnr_disparity"].values[0])
    metrics["fnr_Other"].append(bdf.loc[bdf["attribute_value"]=="Other/Unknown"]["fnr"].values[0])
    metrics["fnrd_Other"].append(bdf.loc[bdf["attribute_value"]=="Other/Unknown"]["fnr_disparity"].values[0])

pd.DataFrame(metrics).to_csv("../experiments_metrics/baseline_logistic_regression_cleanv3.csv")