# Step 13 — Correlation Analysis

# Objective:
# Identify relationships between features and target variable (Survived)

# ------------------------------------------------------------
# 1. CORRELATION HEATMAP
# ------------------------------------------------------------

# Shows linear relationships between all numerical features

# Values range:
# +1 → strong positive
# -1 → strong negative
#  0 → no relationship

# ------------------------------------------------------------
# 2. TOP FEATURES (MOST CORRELATED WITH SURVIVED)
# ------------------------------------------------------------

# 1. sex_encoded (~0.54)
# 2. Pclass (~-0.34)
# 3. Fare (~0.26)
# 4. title-related features
# 5. is_alone / family_size

# Insight:
# Gender is the strongest predictor

# ------------------------------------------------------------
# 3. MULTICOLLINEARITY
# ------------------------------------------------------------

# High correlation (>0.7) between features means redundancy

# Examples:
# - Fare vs fare_per_person
# - family_size vs is_alone

# Risk:
# - Unstable model coefficients
# - Reduced interpretability

# ------------------------------------------------------------
# 4. LOW CORRELATION FEATURES
# ------------------------------------------------------------

# Near zero correlation:
# - SibSp
# - Parch
# - Embarked

# These add little predictive value

# ------------------------------------------------------------
# 🔥 FEATURE SELECTION (VERY IMPORTANT)
# ------------------------------------------------------------

# Best 5 features for Logistic Regression:

# 1. sex_encoded  → strongest predictor
# 2. Pclass       → class-based survival
# 3. Fare         → economic status
# 4. is_alone     → social factor
# 5. age_group_encoded → age-based survival

# ------------------------------------------------------------
# WHY THESE 5?
# ------------------------------------------------------------

# ✔ High correlation with target
# ✔ Low multicollinearity between them
# ✔ Represent different aspects:
#   - Gender
#   - Class
#   - Wealth
#   - Social structure
#   - Age

# ------------------------------------------------------------
# INTERVIEW INSIGHT
# ------------------------------------------------------------

# Q: Why not include all features?

# A:
# - Some are redundant (multicollinearity)
# - Some have low predictive power
# - Simpler models generalize better

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# Selected features provide:
# ✔ Strong predictive power
# ✔ Low redundancy
# ✔ Better model performance

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# SELECT NUMERICAL COLUMNS
# ==============================

num_df = df.select_dtypes(include=['int64', 'float64'])

# ==============================
# CORRELATION MATRIX
# ==============================

corr = num_df.corr()

# ==============================
# HEATMAP (REQUIRED STYLE)
# ==============================

plt.figure(figsize=(14,12))

sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5
)

plt.title("Full Correlation Heatmap")
plt.show()

# ==============================
# CORRELATION WITH SURVIVED
# ==============================

surv_corr = corr['Survived'].sort_values(ascending=False)

print("\n===== FEATURE → SURVIVED CORRELATION =====")
print(surv_corr)

# Top 5 features (excluding Survived)
top5 = surv_corr.drop('Survived').head(5)
print("\n===== TOP 5 FEATURES =====")
print(top5)

# ==============================
# MULTICOLLINEARITY (> 0.7)
# ==============================

high_corr_pairs = []

for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i,j]) > 0.7:
            high_corr_pairs.append(
                (corr.index[i], corr.columns[j], corr.iloc[i,j])
            )

print("\n===== HIGH CORRELATION PAIRS (>0.7) =====")
for pair in high_corr_pairs:
    print(pair)

# ==============================
# LOW CORRELATION (< 0.05)
# ==============================

low_corr = surv_corr[abs(surv_corr) < 0.05]

print("\n===== LOW CORRELATION FEATURES =====")
print(low_corr)