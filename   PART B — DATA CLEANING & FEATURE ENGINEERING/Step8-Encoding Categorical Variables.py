
# Step 8 — Encoding Categorical Variables

# Objective:
# Convert categorical variables into numerical form so they can be used in machine learning models.

# ------------------------------------------------------------
# (a) SEX — LABEL ENCODING
# ------------------------------------------------------------

# Mapping:
# male = 0
# female = 1

# Why this works:
# - Only two categories (binary variable)
# - No risk of introducing false relationships

# Advantage:
# ✔ Simple and efficient
# ✔ No increase in dimensionality

# ------------------------------------------------------------
# (b) EMBARKED — ONE-HOT ENCODING
# ------------------------------------------------------------

# Created columns:
# Embarked_C, Embarked_Q, Embarked_S

# Why One-Hot Encoding?
# - No ordinal relationship between categories
# - Avoids incorrect numeric assumptions

# Example:
# C ≠ 1, Q ≠ 2, S ≠ 3

# ------------------------------------------------------------
# (c) TITLE — ONE-HOT ENCODING
# ------------------------------------------------------------

# Applied one-hot encoding
# Dropped most common category (e.g., Mr)

# Why drop one category?
# - Prevents dummy variable trap
# - Avoids multicollinearity

# ------------------------------------------------------------
# (d) AGE_GROUP & FARE_BIN — ORDINAL ENCODING
# ------------------------------------------------------------

# Mapping:
# Child = 0, Teen = 1, Adult = 2, Senior = 3

# Why ordinal encoding?
# - Categories have natural order
# - Preserves ranking information

# Same for fare_bin:
# Low < Medium < High < VHigh

# ------------------------------------------------------------
# SHAPE ANALYSIS
# ------------------------------------------------------------

# New columns were added due to:
# - One-hot encoding
# - New encoded features

# This increases feature space but improves model understanding

# ------------------------------------------------------------
# FINAL ML FEATURES
# ------------------------------------------------------------

# Selected features include:
# - Numerical features
# - Engineered features
# - Encoded categorical features

# ------------------------------------------------------------
# WHY ENCODING IS IMPORTANT
# ------------------------------------------------------------

# Machine learning models cannot understand text
# Encoding converts categories into numbers

# ------------------------------------------------------------
# INTERVIEW INSIGHT
# ------------------------------------------------------------

# Q: Why not label encode all columns?
# A: Because it introduces false order in non-ordinal data

# Q: Why one-hot encoding?
# A: It preserves independence between categories

# Q: What is dummy variable trap?
# A: Multicollinearity caused by redundant dummy variables

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# Encoding transforms categorical data into a machine-readable format,
# making the dataset fully ready for modeling.


import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# PREPROCESSING (REQUIRED)
# ==============================

# Handle missing values
df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Feature engineering (Step 7 essentials)
df['family_size'] = df['SibSp'] + df['Parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['fare_per_person'] = df['Fare'] / df['family_size']

df['title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
df['title'] = df['title'].apply(lambda x: x if x in common_titles else 'Rare')

def age_group(age):
    if age < 12:
        return 'Child'
    elif age < 18:
        return 'Teen'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

df['age_group'] = df['Age'].apply(age_group)

df['fare_bin'] = pd.qcut(df['Fare'], q=4, labels=['Low','Medium','High','VHigh'])

# ==============================
# STORE SHAPE BEFORE ENCODING
# ==============================

shape_before = df.shape

# ==============================
# (a) SEX — LABEL ENCODING
# ==============================

df['sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})

# ==============================
# (b) EMBARKED — ONE-HOT ENCODING
# ==============================

embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')

df = pd.concat([df, embarked_dummies], axis=1)

# ==============================
# (c) TITLE — ONE-HOT (DROP MOST COMMON)
# ==============================

# Find most common title
most_common_title = df['title'].value_counts().idxmax()

title_dummies = pd.get_dummies(df['title'], prefix='title')

# Drop most common to avoid dummy trap
title_dummies.drop(columns=[f"title_{most_common_title}"], inplace=True)

df = pd.concat([df, title_dummies], axis=1)

# ==============================
# (d) AGE_GROUP & FARE_BIN — ORDINAL ENCODING
# ==============================

age_map = {'Child': 0, 'Teen': 1, 'Adult': 2, 'Senior': 3}
fare_map = {'Low': 0, 'Medium': 1, 'High': 2, 'VHigh': 3}

df['age_group_encoded'] = df['age_group'].map(age_map)
df['fare_bin_encoded'] = df['fare_bin'].map(fare_map)

# ==============================
# FINAL SHAPE AFTER ENCODING
# ==============================

shape_after = df.shape

print("Shape Before Encoding:", shape_before)
print("Shape After Encoding:", shape_after)
print("New Columns Added:", shape_after[1] - shape_before[1])

# ==============================
# ML FEATURE COLUMNS
# ==============================

features = [
    'Pclass', 'sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
    'family_size', 'is_alone', 'fare_per_person',
    'age_group_encoded', 'fare_bin_encoded'
] + list(embarked_dummies.columns) + list(title_dummies.columns)

print("\n===== ML FEATURES =====")
print(features)