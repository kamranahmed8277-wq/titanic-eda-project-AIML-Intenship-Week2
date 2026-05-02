# Step 11 — Advanced Aggregation (agg + transform)

# Objective:
# Perform statistical aggregation and create group-level features for modeling.

# ------------------------------------------------------------
# 1. FARE ANALYSIS (agg function)
# ------------------------------------------------------------

# Metrics computed:
# - mean → average fare per class
# - median → central fare value
# - std → variation in fares
# - min → lowest fare
# - max → highest fare
# - % fares > 50 → premium passengers percentage

# Insight:
# Higher classes (Pclass=1) pay significantly more fares

# ------------------------------------------------------------
# 2. AGE ANALYSIS
# ------------------------------------------------------------

# Metrics:
# - mean age per class
# - median age per class
# - IQR (Q3 - Q1)

# Insight:
# Age distribution varies across passenger classes

# ------------------------------------------------------------
# 3. SURVIVAL ANALYSIS
# ------------------------------------------------------------

# Metrics:
# - survival_rate → mean of Survived
# - total_passengers → count per class

# Insight:
# Survival is strongly linked to class

# ------------------------------------------------------------
# 4. TRANSFORM FUNCTION (IMPORTANT CONCEPT)
# ------------------------------------------------------------

# transform() broadcasts group statistics back to each row

# Created features:
# class_avg_fare → mean fare of passenger's class
# class_survival_rate → survival probability of that class

# Why transform is important:
# - Keeps dataset shape unchanged
# - Adds group context to each row
# - Useful for ML models

# ------------------------------------------------------------
# 5. FINAL COMBINED TABLE
# ------------------------------------------------------------

# Combines:
# - Fare statistics
# - Age statistics
# - Survival statistics

# Gives a full statistical summary per Pclass

# ------------------------------------------------------------
# 6. INTERVIEW INSIGHT
# ------------------------------------------------------------

# Q: Difference between agg and transform?

# A:
# agg → reduces data (summary table)
# transform → keeps same shape (feature engineering)

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# This step converts raw data into:
# ✔ Statistical insights
# ✔ Group-level features
# ✔ ML-ready enriched dataset

import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# BASIC CLEANING (MINIMUM REQUIRED)
# ==============================

df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass','Sex'])['Age'].transform('median')
)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ==============================
# (1) GROUPBY AGG FOR FARE
# ==============================

fare_agg = df.groupby('Pclass')['Fare'].agg([
    'mean',
    'median',
    'std',
    'min',
    'max',
    lambda x: (x > 50).mean() * 100   # % of fares > 50
])

fare_agg = fare_agg.rename(columns={'<lambda_0>': 'fare_%_above_50'})

# ==============================
# (2) GROUPBY AGG FOR AGE
# ==============================

age_agg = df.groupby('Pclass')['Age'].agg([
    'mean',
    'median',
    lambda x: x.quantile(0.75) - x.quantile(0.25)  # IQR
])

age_agg = age_agg.rename(columns={'<lambda_0>': 'age_IQR'})

# ==============================
# (3) SURVIVED STATS
# ==============================

survival_agg = df.groupby('Pclass')['Survived'].agg([
    'mean',
    'count'
])

survival_agg = survival_agg.rename(columns={
    'mean': 'survival_rate',
    'count': 'total_passengers'
})

# ==============================
# (4) COMBINE ALL TABLES
# ==============================

final_agg = pd.concat([fare_agg, age_agg, survival_agg], axis=1)

print("===== FINAL AGGREGATED TABLE =====")
print(final_agg)

# ==============================
# (5) TRANSFORM — BROADCAST FEATURES
# ==============================

df['class_avg_fare'] = df.groupby('Pclass')['Fare'].transform('mean')
df['class_survival_rate'] = df.groupby('Pclass')['Survived'].transform('mean')

# ==============================
# (6) DISPLAY FINAL DATA
# ==============================

print("\n===== FIRST 15 ROWS WITH NEW FEATURES =====")
print(df[['Pclass', 'Fare', 'class_avg_fare', 'class_survival_rate']].head(15))