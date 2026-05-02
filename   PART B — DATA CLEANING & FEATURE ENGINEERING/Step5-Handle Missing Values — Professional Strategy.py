import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# 1. AGE — GROUP-LEVEL IMPUTATION
# ==============================

# Fill Age using median grouped by Pclass and Sex
df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
)

# ==============================
# 2. EMBARKED — MODE IMPUTATION
# ==============================

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ==============================
# 3. CABIN — FEATURE ENGINEERING + DROP
# ==============================

# Create binary feature: 1 if cabin exists, else 0
df['has_cabin'] = df['Cabin'].notnull().astype(int)

# Drop original Cabin column
df.drop(columns=['Cabin'], inplace=True)

# ==============================
# 4. FINAL CHECK — NO MISSING VALUES
# ==============================

print("===== FINAL MISSING VALUES CHECK =====")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

# ==============================
# 5. CLEANING SUMMARY
# ==============================

print("\n===== CLEANING SUMMARY =====")
print("Age      → Filled using median grouped by Pclass & Sex")
print("Embarked → Filled using mode")
print("Cabin    → Converted to 'has_cabin' + original column dropped")


# Step 5 — Handle Missing Values (Professional Strategy)

## Objective
#Handle missing values using domain-aware and statistically sound techniques.

#---

## 1. Age (177 Missing Values)

### Strategy Used:
#Filled missing values using **median grouped by Pclass and Sex**:

#```python
#df.groupby(['Pclass','Sex'])['Age'].transform('median')