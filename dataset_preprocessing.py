import os
import sys

import numpy as np
import scipy
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import GroupImputer

dictionary = pd.read_csv("WiDS2021/DataDictionaryWiDS2021.csv")
dataset = pd.read_csv("WiDS2021/TrainingWiDS2021.csv", index_col=0)
y = dataset["diabetes_mellitus"]

# Ids features
ids = ["encounter_id", "icu_id", "hospital_id"]
dataset = dataset.drop(ids, axis=1)

# Zero variance features
zero_variance = ["readmission_status"]
dataset = dataset.drop(zero_variance, axis=1)

# More-than-80% missing values features
perc = 80.0
min_count = int(((100-perc)/100)*dataset.shape[0] + 1)
dataset = dataset.dropna(axis=1, thresh=min_count)

# See docs
apache_diagnosis = ["apache_2_diagnosis", "apache_3j_diagnosis"]
dataset = dataset.drop(apache_diagnosis, axis=1)

# Collinear features (cor > 0.95)
corr_matrix = dataset.corr().abs()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
dataset = dataset.drop(to_drop, axis=1)

# Get feature type from dict for imputation statistic
dictionary = dictionary[dictionary["Variable Name"].isin(dataset.columns.to_list())]
nums = dictionary[dictionary["Data Type"]=="numeric"]["Variable Name"]
bins = dictionary[dictionary["Data Type"]=="binary"]["Variable Name"]
ords = dictionary[dictionary["Data Type"]=="integer"]["Variable Name"]
strs = dictionary[dictionary["Data Type"]=="string"]["Variable Name"]

# Create groups
dataset["age_grouped"] = pd.cut(
    dataset["age"],
    bins=[-np.inf, 25, 65, np.inf],
    include_lowest=True,
    labels=["Youth", "Middle-age", "Elderly"]
)

dataset["bmi_grouped"] = pd.cut(
    dataset["bmi"],
    bins=[-np.inf, 18.5, 25, 30, np.inf],
    include_lowest=True,
    labels=["Underwieght", "Normal", "Overwieght", "Obese"]
)

has_demographics = dataset["gender"].isna() | dataset["bmi"].isna() | dataset["age"].isna()

with_demographics = dataset[~has_demographics]
without_demographics = dataset[has_demographics]

# Group-specific median imputation for numerical features
for col in nums:
    imputer = GroupImputer(["gender", "age_grouped", "bmi_grouped"], col, metric="median")
    imputer.fit(with_demographics)
    with_demographics = imputer.transform(with_demographics)

# Group-specific mode imputation for binary features
for col in bins:
    imputer = GroupImputer(["gender", "age_grouped", "bmi_grouped"], col, metric="median")
    imputer.fit(with_demographics)
    with_demographics = imputer.transform(with_demographics)

# Group-specific median imputation for ordinal features
for col in ords:
    imputer = GroupImputer(["gender", "age_grouped", "bmi_grouped"], col, metric="median")
    imputer.fit(with_demographics)
    with_demographics = imputer.transform(with_demographics)

# If demogragphics is missing, impute with entire dataset
for col in list(bins) + list(nums) + list(ords):
    without_demographics[col] = without_demographics[col].fillna(dataset[col].median())

dataset = pd.concat([with_demographics, without_demographics]).sort_index()

# Discard category imputation for categorical features
dataset["ethnicity"] = dataset["ethnicity"].fillna("Other/Unknown")
dataset["hospital_admit_source"] = dataset["hospital_admit_source"].fillna("Other")
dataset["icu_admit_source"] = dataset["icu_admit_source"].fillna("Other/Unknown")
dataset["gender"] = dataset["gender"].fillna("Other")

protected_attributes = dataset[["ethnicity", "gender"]]
protected_attributes.to_csv("WiDS2021/protected_attributes.csv", index=False)

# Onehot encoding
onehot_dataset = pd.get_dummies(dataset, drop_first=True)

# Scale data
scaler = StandardScaler()
array_scaled_dataset = scaler.fit_transform(onehot_dataset)

scaled_dataset = pd.DataFrame(
    array_scaled_dataset,
    index=onehot_dataset.index,
    columns=onehot_dataset.columns
)

scaled_dataset["diabetes_mellitus"] = y

grouped_cols = [col for col in scaled_dataset.columns if 'grouped' in col]
scaled_dataset = scaled_dataset.drop(grouped_cols, axis=1)

scaled_dataset.to_csv("WiDS2021/wids2021_dataset_cleaned_v3.csv", index=False)




