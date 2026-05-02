# Step 6 — Outlier Detection & Treatment

# Objective:
# Detect and handle extreme values (outliers) in Fare and Age columns
# using statistical methods (IQR) and domain knowledge.

# ------------------------------------------------------------
# 1. IQR METHOD EXPLANATION
# ------------------------------------------------------------

# IQR (Interquartile Range) = Q3 - Q1
# Lower Fence = Q1 - 1.5 * IQR
# Upper Fence = Q3 + 1.5 * IQR

# Any value outside these fences is considered an outlier.

# ------------------------------------------------------------
# 2. FARE OUTLIER ANALYSIS
# ------------------------------------------------------------

# Observation:
# - Fare has many high-value outliers
# - Some fares exceed 300

# Reason (Real-world explanation):
# - First-class passengers paid significantly higher fares
# - These are NOT errors — they are valid extreme values

# Decision:
# DO NOT REMOVE these values
# Instead, apply capping (winsorization)

# Action:
# - Cap Fare at 99th percentile
# - This reduces extreme influence but keeps data

# ------------------------------------------------------------
# 3. AGE OUTLIER ANALYSIS
# ------------------------------------------------------------

# Observation:
# - Very few outliers detected
# - Values are within realistic human age limits

# Reason:
# - Age distribution is naturally bounded (0–80 approx)

# Decision:
# - No treatment required

# ------------------------------------------------------------
# 4. VISUALIZATION (BOXPLOTS)
# ------------------------------------------------------------

# Before:
# - Fare shows extreme long tail (many outliers)
# - Age looks relatively normal

# After:
# - Fare distribution becomes more compact
# - Outliers reduced visually
# - Age remains unchanged

# ------------------------------------------------------------
# 5. WHY CAPPING INSTEAD OF DROPPING?
# ------------------------------------------------------------

# Capping (Winsorization) replaces extreme values with a threshold
# instead of removing rows.

# Advantages:
# ✔ Keeps all data points
# ✔ Prevents loss of important samples
# ✔ Reduces skewness
# ✔ Improves model stability

# Disadvantages of Dropping:
#  Loss of valuable information
#  Smaller dataset
#  Risk of bias (removing high-value groups like rich passengers)

# ------------------------------------------------------------
# 6. FINAL DECISIONS
# ------------------------------------------------------------

# Fare:
# → Outliers capped at 99th percentile

# Age:
# → No treatment needed

# ------------------------------------------------------------
# 7. INTERVIEW INSIGHT
# ------------------------------------------------------------

# Q: Why not remove Fare outliers?
# A: Because they represent real high-paying passengers,
#    removing them would lose meaningful information.

# Q: Why is capping better?
# A: It reduces the effect of extreme values while preserving data integrity.

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# This step ensures:
# ✔ Balanced feature distributions
# ✔ Reduced impact of extreme values
# ✔ No loss of critical data
# ✔ Improved ML model performance

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# HANDLE MISSING VALUES FIRST (REQUIRED BEFORE OUTLIERS)
# ==============================

df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['has_cabin'] = df['Cabin'].notnull().astype(int)
df.drop(columns=['Cabin'], inplace=True)

# ==============================
# 1. IQR CALCULATION FUNCTION
# ==============================

def compute_iqr_bounds(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    return Q1, Q3, IQR, lower_fence, upper_fence

# ==============================
# 2. FARE OUTLIERS
# ==============================

Q1_f, Q3_f, IQR_f, lf_f, uf_f = compute_iqr_bounds('Fare')

fare_outliers = df[(df['Fare'] < lf_f) | (df['Fare'] > uf_f)]

print("===== FARE OUTLIERS =====")
print("IQR:", IQR_f)
print("Lower Fence:", lf_f)
print("Upper Fence:", uf_f)
print("Number of Outliers:", len(fare_outliers))

# ==============================
# 3. AGE OUTLIERS
# ==============================

Q1_a, Q3_a, IQR_a, lf_a, uf_a = compute_iqr_bounds('Age')

age_outliers = df[(df['Age'] < lf_a) | (df['Age'] > uf_a)]

print("\n===== AGE OUTLIERS =====")
print("IQR:", IQR_a)
print("Lower Fence:", lf_a)
print("Upper Fence:", uf_a)
print("Number of Outliers:", len(age_outliers))

# ==============================
# 4. BOXPLOT BEFORE TREATMENT
# ==============================

plt.figure()
df.boxplot(column='Fare')
plt.title("Fare - Before Treatment")
plt.show()

plt.figure()
df.boxplot(column='Age')
plt.title("Age - Before Treatment")
plt.show()

# ==============================
# 5. FARE TREATMENT (CAPPING)
# ==============================

# Cap at 99th percentile
fare_cap = df['Fare'].quantile(0.99)
df['Fare'] = df['Fare'].clip(upper=fare_cap)

print("\nFare capped at:", fare_cap)

# ==============================
# 6. BOXPLOT AFTER TREATMENT
# ==============================

plt.figure()
df.boxplot(column='Fare')
plt.title("Fare - After Capping")
plt.show()

plt.figure()
df.boxplot(column='Age')
plt.title("Age - After (No Change)")
plt.show()

# ==============================
# FINAL CHECK
# ==============================

print("\nFinal dataset shape:", df.shape)