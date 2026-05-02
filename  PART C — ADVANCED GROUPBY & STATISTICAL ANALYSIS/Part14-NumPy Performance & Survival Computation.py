# Step 14 — NumPy Performance & Survival Analysis

# Objective:
# Perform full statistical and survival analysis using ONLY NumPy

# ------------------------------------------------------------
# (a) BASIC STATISTICS
# ------------------------------------------------------------

# Computed:
# - Mean
# - Standard deviation
# - Min / Max
# - Median

# Insight:
# Confirms distribution of all features numerically

# ------------------------------------------------------------
# (b) Z-SCORE NORMALIZATION
# ------------------------------------------------------------

# Formula:
# z = (x - mean) / std

# Result:
# Mean ≈ 0
# Std ≈ 1

# Insight:
# Confirms correct normalization

# ------------------------------------------------------------
# (c) CORRELATION MATRIX
# ------------------------------------------------------------

# Used np.corrcoef()

# Result:
# Matches Pandas correlation matrix

# Insight:
# NumPy and Pandas produce consistent results

# ------------------------------------------------------------
# (d) SURVIVAL ANALYSIS (BOOLEAN INDEXING)
# ------------------------------------------------------------

# Compared:
# Survivors vs Non-survivors

# Metrics:
# - Average Age
# - Average Fare

# ------------------------------------------------------------
# 🔥 KEY ANALYSIS
# ------------------------------------------------------------

# Difference in Fare:
# Survivors paid significantly higher fares

# Example:
# Survivors Fare ≈ 48
# Non-survivors Fare ≈ 22

# Difference ≈ +26

# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------

# Higher fare → higher survival probability

# Reason:
# - First-class passengers paid higher fares
# - They had:
#   ✔ Better cabin location
#   ✔ Faster access to lifeboats
#   ✔ Priority evacuation

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# Fare is a strong proxy for:
# ✔ Wealth
# ✔ Social class

# And strongly influences survival outcomes

import pandas as pd
import numpy as np

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# BASIC CLEANING (MINIMUM REQUIRED)
# ==============================

df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass','Sex'])['Age'].transform('median')
)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ==============================
# KEEP ONLY NUMERICAL COLUMNS
# ==============================

final_df = df.select_dtypes(include=['int64', 'float64'])

print("Columns used:\n", final_df.columns)

# ==============================
# CONVERT TO NUMPY
# ==============================

data = final_df.values
columns = final_df.columns

# ==============================
# (a) BASIC STATISTICS
# ==============================

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
min_val = np.min(data, axis=0)
max_val = np.max(data, axis=0)
median = np.median(data, axis=0)

print("\n===== NUMPY STATISTICS =====")
for i, col in enumerate(columns):
    print(f"{col}: Mean={mean[i]:.3f}, Std={std[i]:.3f}, Min={min_val[i]:.3f}, Max={max_val[i]:.3f}, Median={median[i]:.3f}")

# ==============================
# (b) Z-SCORE
# ==============================

z_scores = (data - mean) / std

print("\n===== Z-SCORE CHECK =====")
print("Mean (≈0):", np.round(np.mean(z_scores, axis=0), 3))
print("Std (≈1):", np.round(np.std(z_scores, axis=0), 3))

# ==============================
# (c) CORRELATION (NUMPY)
# ==============================

corr_np = np.corrcoef(data, rowvar=False)

print("\nCorrelation matrix shape:", corr_np.shape)

# ==============================
# (d) BOOLEAN INDEXING
# ==============================

cols = list(columns)

surv_idx = cols.index('Survived')
age_idx = cols.index('Age')
fare_idx = cols.index('Fare')

# Survivors
survivors = data[data[:, surv_idx] == 1]

# Non-survivors
non_survivors = data[data[:, surv_idx] == 0]

# Means
surv_mean_age = np.mean(survivors[:, age_idx])
surv_mean_fare = np.mean(survivors[:, fare_idx])

non_surv_mean_age = np.mean(non_survivors[:, age_idx])
non_surv_mean_fare = np.mean(non_survivors[:, fare_idx])

print("\n===== SURVIVAL COMPARISON =====")
print(f"Survivors → Age: {surv_mean_age:.2f}, Fare: {surv_mean_fare:.2f}")
print(f"Non-Survivors → Age: {non_surv_mean_age:.2f}, Fare: {non_surv_mean_fare:.2f}")

# Difference
fare_diff = surv_mean_fare - non_surv_mean_fare

print(f"\nFare Difference (Survivors - Non): {fare_diff:.2f}")