# Titanic Dataset — Column Description

#This dataset contains passenger information from the Titanic disaster and is used to predict survival.

#---

## Column Explanation

#1. **PassengerId (int64)**
 #  Unique identifier for each passenger.

#2. **Survived (int64)**
   #Target variable:
   #- 0 = Did not survive
   #- 1 = Survived

#3. **Pclass (int64)**
   #Passenger class:
   #- 1 = First class
   #- 2 = Second class
   #- 3 = Third class
   #Represents socioeconomic status.

#4. **Name (object)**
  # Full name of the passenger. Mostly useful for feature engineering.

#5. **Sex (object)**
   #Gender of the passenger (male/female).

#6. **Age (float64)**
   #Age of the passenger. Contains missing values.

#7. **SibSp (int64)**
   #Number of siblings/spouses aboard the Titanic.

#8. **Parch (int64)**
   #Number of parents/children aboard the Titanic.

#9. **Ticket (object)**
   #Ticket number. Often contains mixed formats.

#10. **Fare (float64)**
   #Ticket price paid by the passenger.

#11. **Cabin (object)**
   #Cabin number. Contains a large number of missing values.

#12. **Embarked (object)**
   #Port of embarkation:
   #- C = Cherbourg
   #- Q = Queenstown
   #- S = Southampton

#---

## Summary Insights

#- Dataset contains **891 rows and 12 columns**.
#- Mix of **categorical and numerical features**.
#- Missing values present in:
  #- Age
  #- Cabin (heavily missing)
 # - Embarked
#- Target variable is **Survived**.
# Some columns (Name, Ticket, Cabin) may need feature engineering or removal.

#---

## Next Step

#Handle missing values and prepare data for modeling.

import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# BASIC DATA PREVIEW
# ==============================

print("===== HEAD (First 10 Rows) =====")
print(df.head(10))

print("\n===== TAIL (Last 5 Rows) =====")
print(df.tail(5))

print("\n===== RANDOM SAMPLE (8 Rows) =====")
print(df.sample(8, random_state=42))

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())


# ==============================
# COLUMN TYPE COUNTS
# ==============================

categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).shape[1]
numerical_cols = df.select_dtypes(exclude=['object', 'string', 'category']).shape[1]

print("\n===== COLUMN TYPE COUNTS =====")
print("Categorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)


# ==============================
# MISSING VALUES ANALYSIS
# ==============================

missing_columns = df.isnull().any().sum()
total_missing_cells = df.isnull().sum().sum()

print("\n===== MISSING VALUES =====")
print("Columns with ANY missing values:", missing_columns)
print("Total missing cells in dataset:", total_missing_cells)