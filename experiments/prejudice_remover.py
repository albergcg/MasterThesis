import os
import sys
from tqdm import tqdm

import numpy as np
import scipy
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import PrejudiceRemover

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness

from warnings import simplefilter
simplefilter(action='ignore')

g = Group()
b = Bias()

icu_dataset = pd.read_csv("../WiDS2021/wids2021_dataset_cleaned_v3.csv")
protected_attributes = pd.read_csv("../WiDS2021/protected_attributes.csv")["ethnicity"]

encoder = LabelEncoder()
icu_dataset["eth"] = encoder.fit_transform(protected_attributes)

eth_dummy_cols = [col for col in icu_dataset.columns if 'ethnicity' in col]
icu_dataset = icu_dataset.drop(eth_dummy_cols, axis=1)

X = icu_dataset.drop("diabetes_mellitus", axis=1)
y = icu_dataset["diabetes_mellitus"].rename("label_value")

n_cpu = os.cpu_count()

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

    for dev_idx, test_idx in tqdm(cv_outer.split(X, y), leave=False):
        X_dev, X_test = X.loc[dev_idx], X.loc[test_idx]
        y_dev, y_test = y.loc[dev_idx], y.loc[test_idx]
    
        dev_merged = pd.concat([X_dev, y_dev], axis=1)
        test_merged = pd.concat([X_test, y_test], axis=1)

        dev_aif360 = BinaryLabelDataset(
            label_names=["label_value"],
            protected_attribute_names=["eth"],
            df=dev_merged
        )

        test_aif360 = BinaryLabelDataset(
            label_names=["label_value"],
            protected_attribute_names=["eth"],
            df=test_merged
        )

        model = PrejudiceRemover()
        model.fit(dev_aif360)
        yhat = model.predict(test_aif360).scores.reshape(-1)
        yhat_index = pd.Series(yhat, index=y_test.index).rename("score")
        yhats = pd.concat([yhats, yhat_index])

    yhats = np.round(yhats.sort_index().rename("score"))
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

pd.DataFrame(metrics).to_csv("../experiments_metrics/prejudice_remover_cleanv3.csv")