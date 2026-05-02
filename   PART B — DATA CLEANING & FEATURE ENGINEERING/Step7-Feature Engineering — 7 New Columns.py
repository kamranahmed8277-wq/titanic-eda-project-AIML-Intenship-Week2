# Step 7 — Feature Engineering (7 New Columns)

# Objective:
# Create new meaningful features from existing data to improve model performance.

# ------------------------------------------------------------
# (a) FAMILY SIZE
# ------------------------------------------------------------

# Formula:
# family_size = SibSp + Parch + 1

# Explanation:
# - SibSp = siblings/spouses
# - Parch = parents/children
# - +1 includes the passenger

# Insight:
# Family size affects survival probability

# ------------------------------------------------------------
# (b) IS ALONE
# ------------------------------------------------------------

# Logic:
# If family_size == 1 → passenger is alone

# Insight:
# People traveling alone had different survival chances

# ------------------------------------------------------------
# (c) FARE PER PERSON
# ------------------------------------------------------------

# Formula:
# Fare / family_size

# Insight:
# Reflects actual cost per individual instead of total ticket fare

# Helps normalize fare differences between families

# ------------------------------------------------------------
# (d) TITLE
# ------------------------------------------------------------

# Extracted from Name column

# Examples:
# Mr, Mrs, Miss, Master

# Rare titles grouped into 'Rare'

# Insight:
# Title strongly correlates with:
# - Gender
# - Age
# - Social status

# ------------------------------------------------------------
# (e) AGE GROUP
# ------------------------------------------------------------

# Categories:
# Child (<12)
# Teen (12–17)
# Adult (18–60)
# Senior (60+)

# Insight:
# Survival patterns differ by age group

# ------------------------------------------------------------
# (f) DECK
# ------------------------------------------------------------

# Extracted first letter from Cabin

# Missing values → 'Unknown'

# Insight:
# Deck location relates to:
# - Passenger class
# - Proximity to lifeboats

# ------------------------------------------------------------
# (g) FARE BIN
# ------------------------------------------------------------

# Used pd.qcut to divide Fare into 4 equal groups

# Labels:
# Low, Medium, High, VHigh

# Insight:
# Converts continuous variable into categorical groups

# ------------------------------------------------------------
# FINAL FEATURE SUMMARY
# ------------------------------------------------------------

# family_size      → numerical
# is_alone         → binary
# fare_per_person  → numerical
# title            → categorical
# age_group        → categorical
# deck             → categorical
# fare_bin         → categorical

# ------------------------------------------------------------
# WHY FEATURE ENGINEERING IS IMPORTANT
# ------------------------------------------------------------

# ✔ Improves model accuracy
# ✔ Captures hidden patterns
# ✔ Adds domain knowledge
# ✔ Transforms raw data into meaningful signals

# ------------------------------------------------------------
# INTERVIEW INSIGHT
# ------------------------------------------------------------

# Q: Why create new features instead of using raw data?
# A: Because raw data may not directly represent patterns;
#    engineered features expose relationships that models can learn.

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# This step enhances dataset quality and predictive power,
# making it suitable for advanced machine learning models.


import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# HANDLE MISSING VALUES (REQUIRED BEFORE FEATURE ENGINEERING)
# ==============================

df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
)

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Keep Cabin copy before dropping
df['deck'] = df['Cabin'].str[0]

df['has_cabin'] = df['Cabin'].notnull().astype(int)

# ==============================
# (a) FAMILY SIZE
# ==============================

df['family_size'] = df['SibSp'] + df['Parch'] + 1

# ==============================
# (b) IS ALONE
# ==============================

df['is_alone'] = (df['family_size'] == 1).astype(int)

# ==============================
# (c) FARE PER PERSON
# ==============================

df['fare_per_person'] = df['Fare'] / df['family_size']

# ==============================
# (d) TITLE EXTRACTION
# ==============================

df['title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Group rare titles
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']

df['title'] = df['title'].apply(lambda x: x if x in common_titles else 'Rare')

# ==============================
# (e) AGE GROUP
# ==============================

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

# ==============================
# (f) DECK (FROM CABIN)
# ==============================

# Fill missing deck as 'Unknown'
df['deck'] = df['deck'].fillna('Unknown')

# ==============================
# (g) FARE BIN
# ==============================

df['fare_bin'] = pd.qcut(
    df['Fare'],
    q=4,
    labels=['Low', 'Medium', 'High', 'VHigh']
)

# ==============================
# DROP ORIGINAL CABIN
# ==============================

df.drop(columns=['Cabin'], inplace=True)

# ==============================
# VALUE COUNTS FOR CATEGORICAL FEATURES
# ==============================

print("===== VALUE COUNTS =====")

print("\nTitle:\n", df['title'].value_counts())
print("\nAge Group:\n", df['age_group'].value_counts())
print("\nDeck:\n", df['deck'].value_counts())
print("\nFare Bin:\n", df['fare_bin'].value_counts())

# ==============================
# DISPLAY FIRST 10 ROWS (NEW FEATURES)
# ==============================

print("\n===== FIRST 10 ROWS (NEW FEATURES) =====")
print(df[['family_size', 'is_alone', 'fare_per_person',
          'title', 'age_group', 'deck', 'fare_bin']].head(10))