# Missing Value Deep Analysis — Titanic Dataset

## Overview

#The dataset contains missing values in the following columns:

#- Cabin (~77% missing)
#- Age (~20% missing)
#- Embarked (<1% missing)

#---

## 1. Cabin (High Missing — ~77%)

### (a) Why missing?
#In a real shipwreck scenario:
#- Not all passengers were assigned cabins (especially 3rd class).
#- Records may have been lost during the disaster.
#- Lower-class passengers often traveled in shared or undocumented spaces.

### (b) Best Strategy
# Drop OR create a new feature

#Options:
#- Drop the column (too many missing values)
#- OR extract deck info (first letter like C, B, E)

###  Recommended:
#Drop the column OR convert to a binary feature:
#```python
#df['HasCabin'] = df['Cabin'].notnull().astype(int)

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# MISSING VALUES COUNT
# ==============================

print("===== MISSING VALUES (COUNT) =====")
missing_counts = df.isnull().sum()
print(missing_counts)

# ==============================
# MISSING PERCENTAGE
# ==============================

print("\n===== MISSING VALUES (PERCENTAGE) =====")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent)

# ==============================
# VISUALIZATION
# ==============================

print("\nDisplaying Missing Data Visualization...")
msno.bar(df)
plt.show()

# ==============================
# TOP 3 MISSING COLUMNS
# ==============================

top_missing = missing_counts.sort_values(ascending=False).head(3)

print("\n===== TOP 3 COLUMNS WITH MOST MISSING VALUES =====")
print(top_missing)