# Step 9 — Feature Scaling & Final ML-Ready Dataset

# Objective:
# Normalize numerical features and prepare the final dataset for machine learning.

# ------------------------------------------------------------
# 1. FEATURE SCALING (StandardScaler)
# ------------------------------------------------------------

# Applied StandardScaler to:
# - Age
# - Fare
# - fare_per_person
# - family_size

# Formula:
# z = (x - mean) / std

# ------------------------------------------------------------
# 2. WHY SCALING IS IMPORTANT
# ------------------------------------------------------------

# Machine learning models are sensitive to feature magnitude

# Example:
# Fare values (~0–500) vs Age (~0–80)

# Without scaling:
# - Larger values dominate the model

# With scaling:
# - All features contribute equally

# ------------------------------------------------------------
# 3. BEFORE vs AFTER SCALING
# ------------------------------------------------------------

# Before:
# Mean ≠ 0
# Std ≠ 1

# After:
# Mean ≈ 0
# Std ≈ 1

# This confirms proper normalization

# ------------------------------------------------------------
# 4. IMPORTANT RULE
# ------------------------------------------------------------

# Always fit scaler ONLY on training data

# In this task:
# - Full dataset treated as training data
# - So we used fit_transform()

# ------------------------------------------------------------
# 5. FINAL DATASET CREATION
# ------------------------------------------------------------

# Dropped unnecessary columns:
# - Name (text, not useful directly)
# - Ticket (high cardinality)
# - PassengerId (identifier only)

# Remaining columns = ML features

# ------------------------------------------------------------
# 6. FINAL DATASET STRUCTURE
# ------------------------------------------------------------

# Contains:
# ✔ Scaled numerical features
# ✔ Encoded categorical features
# ✔ Engineered features

# Dataset is now:
# ✔ Clean
# ✔ Structured
# ✔ Ready for ML models

# ------------------------------------------------------------
# 7. SAVING DATASET
# ------------------------------------------------------------

# Saved as:
# titanic_cleaned.csv

# This file can be used directly for:
# - Model training
# - Cross-validation
# - Deployment

# ------------------------------------------------------------
# INTERVIEW INSIGHT
# ------------------------------------------------------------

# Q: Why use StandardScaler?
# A: It standardizes features to mean 0 and std 1,
#    improving model performance and convergence.

# Q: What happens if we don't scale?
# A: Features with larger ranges dominate the model.

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# The dataset is now fully ML-ready with:
# ✔ Clean data
# ✔ Engineered features
# ✔ Encoded variables
# ✔ Scaled numerical values



import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# STEP 5 — CLEANING
# ==============================

df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['has_cabin'] = df['Cabin'].notnull().astype(int)
df.drop(columns=['Cabin'], inplace=True)

# ==============================
# STEP 7 — FEATURE ENGINEERING
# ==============================

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
# STEP 8 — ENCODING
# ==============================

df['sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})

embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)

title_dummies = pd.get_dummies(df['title'], prefix='title')
most_common_title = df['title'].value_counts().idxmax()
title_dummies.drop(columns=[f"title_{most_common_title}"], inplace=True)
df = pd.concat([df, title_dummies], axis=1)

age_map = {'Child': 0, 'Teen': 1, 'Adult': 2, 'Senior': 3}
fare_map = {'Low': 0, 'Medium': 1, 'High': 2, 'VHigh': 3}

df['age_group_encoded'] = df['age_group'].map(age_map)
df['fare_bin_encoded'] = df['fare_bin'].map(fare_map)

# ==============================
# STEP 9 — FEATURE SCALING
# ==============================

scaler = StandardScaler()

scale_cols = ['Age', 'Fare', 'fare_per_person', 'family_size']

# BEFORE SCALING
print("===== BEFORE SCALING =====")
for col in scale_cols:
    print(f"{col} → Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")

# APPLY SCALING
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# AFTER SCALING
print("\n===== AFTER SCALING =====")
for col in scale_cols:
    print(f"{col} → Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")

# ==============================
# FINAL DATASET (ML READY)
# ==============================

final_df = df.drop(columns=['Name', 'Ticket', 'PassengerId'])

print("\n===== FINAL DATASET INFO =====")
print(final_df.info())

print("\n===== FINAL DATASET HEAD =====")
print(final_df.head())

# ==============================
# SAVE CLEAN DATASET
# ==============================

final_df.to_csv("titanic_cleaned.csv", index=False)

print("\nDataset saved as 'titanic_cleaned.csv'")