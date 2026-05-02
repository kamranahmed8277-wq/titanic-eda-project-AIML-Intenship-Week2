# Step 4 — Data Type Audit & Correction (Titanic Dataset)

## Objective
#Ensure each column has the correct data type based on its meaning and usage in analysis and machine learning.

#---

## 1. Numerical Columns Stored as Objects

### Observation:
#- No major numeric columns were incorrectly stored as strings.
#- Age and Fare were already numeric but were revalidated using `pd.to_numeric()`.

### Conclusion:
#✔ Dataset is clean in terms of numeric storage
#✔ No major corrections required

#---

## 2. Categorical Columns Conversion

### Converted Columns:
#- Sex → category
#- Embarked → category
#- Pclass → category
#- Survived → category

#---

## 3. Why Convert 'Survived' to Category?

#Although Survived is stored as integers (0/1), it represents:

#- 0 = Did not survive
#- 1 = Survived

# This is NOT a continuous numeric variable
# It is a **binary classification label**

###  Reason:
#- Treating it as category makes it clear it's a **target class**
#- Prevents algorithms from assuming numeric relationships

#---

## 4. Why Convert 'Pclass' to Category?

#Pclass values:
#- 1 = First class
#- 2 = Second class
#- 3 = Third class

# These are **labels**, not actual numbers

###  Problem if left numeric:
#Model may think:
#- Class 3 > Class 1 (which is incorrect)

### Reason:
#- It represents **ordinal categories**
#- Should not be treated as continuous numeric data

#---

## 5. Why NOT Convert Some Columns?

#| Column   | Reason |
#|----------|--------|
#| Name     | High uniqueness (not useful directly) |
#| Ticket   | Mixed format, not structured |
#| Cabin    | Too many missing values |

#---

## 6. Before vs After Comparison

#We compared original and updated data types to ensure correctness.

### Key Changes:
#- Survived → int64 → category
#- Pclass → int64 → category
#- Sex → object → category
#- Embarked → object → category

#---

## 7. Key Benefits of Data Type Correction

#- Improves model understanding
#- Reduces memory usage
#- Prevents incorrect mathematical assumptions
#- Makes feature encoding easier

#---

## Interview Answer

#"Why convert Survived to category?"

#Because it represents a classification label (0 or 1), not a continuous numeric variable.

# "Why convert Pclass to category?"

#Because it represents passenger classes (labels), not actual numeric values with mathematical meaning.



## Final Insight

#Correct data typing is essential for:
#- Accurate analysis
#- Better machine learning performance
#- Avoiding logical errors in modeling











import pandas as pd

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# STORE ORIGINAL DATA TYPES
# ==============================

original_dtypes = df.dtypes.reset_index()
original_dtypes.columns = ['Column', 'Old_Dtype']

print("===== ORIGINAL DATA TYPES =====")
print(original_dtypes)

# ==============================
# 1. FIX NUMERIC COLUMNS (IF NEEDED)
# ==============================

# Ensure numeric columns are properly converted
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

# ==============================
# 2. CONVERT CATEGORICAL COLUMNS
# ==============================

# Explicit categorical conversion
categorical_columns = ['Sex', 'Embarked', 'Pclass', 'Survived']

for col in categorical_columns:
    df[col] = pd.Categorical(df[col])

# ==============================
# STORE UPDATED DATA TYPES
# ==============================

updated_dtypes = df.dtypes.reset_index()
updated_dtypes.columns = ['Column', 'New_Dtype']

# ==============================
# MERGE BEFORE & AFTER
# ==============================

comparison = pd.merge(original_dtypes, updated_dtypes, on='Column')

print("\n===== DATA TYPE COMPARISON =====")
print(comparison)

# ==============================
# FINAL INFO
# ==============================

print("\n===== UPDATED DATASET INFO =====")
df.info()