# Step 12 — Pivot Table Analysis

# Objective:
# Use pivot tables to analyze survival patterns across multiple variables.

# ------------------------------------------------------------
# (a) Pclass vs Sex
# ------------------------------------------------------------

# Observation:
# - First-class females have the highest survival (~97%)
# - Third-class males have the lowest survival (~14%)

# Insight:
# Survival strongly depends on both gender and class

# ------------------------------------------------------------
# (b) Age Group vs Pclass
# ------------------------------------------------------------

# Observation:
# - Children tend to have higher survival rates
# - First-class passengers outperform other classes across all age groups

# Insight:
# Age and class together influence survival chances

# ------------------------------------------------------------
# (c) Title vs Pclass (Fare)
# ------------------------------------------------------------

# Observation:
# - Titles like "Mr" dominate lower classes
# - Higher titles associated with higher fares

# Insight:
# Title reflects social status and economic condition

# ------------------------------------------------------------
# 🔥 ANALYSIS REQUIRED
# ------------------------------------------------------------

# Highest Survival Combination:
# → First-class females (~97%)

# Lowest Survival Combination:
# → Third-class males (~14%)

# ------------------------------------------------------------
# EXPLANATION (REAL-WORLD FACTORS)
# ------------------------------------------------------------

# 1. "Women and children first" policy
# → Women were prioritized for lifeboats

# 2. Social class advantage
# → First-class passengers had:
#   - Better cabin locations
#   - Faster access to lifeboats
#   - More crew assistance

# 3. Third-class disadvantage
# → Located in lower decks
# → Limited access to exits
# → Delayed evacuation

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# Survival was heavily influenced by:
# ✔ Gender (strongest factor)
# ✔ Passenger class (second strongest)

# Combined effect creates extreme survival differences


import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# BASIC PREPROCESSING
# ==============================

df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass','Sex'])['Age'].transform('median')
)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Age group
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

# Title
df['title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
df['title'] = df['title'].apply(lambda x: x if x in common_titles else 'Rare')

# ==============================
# (a) Pclass vs Sex (Survival Rate)
# ==============================

pivot_a = pd.pivot_table(df, values='Survived',
                         index='Pclass', columns='Sex',
                         aggfunc='mean')

print("\nPivot (a):\n", pivot_a.round(3))

# Heatmap
plt.figure()
plt.imshow(pivot_a)
plt.xticks(range(len(pivot_a.columns)), pivot_a.columns)
plt.yticks(range(len(pivot_a.index)), pivot_a.index)

for i in range(len(pivot_a.index)):
    for j in range(len(pivot_a.columns)):
        plt.text(j, i, f"{pivot_a.iloc[i,j]:.2f}",
                 ha='center', va='center')

plt.title("Survival Rate: Pclass vs Sex")
plt.colorbar()
plt.show()

# ==============================
# (b) Age Group vs Pclass
# ==============================

pivot_b = pd.pivot_table(df, values='Survived',
                         index='age_group', columns='Pclass',
                         aggfunc=['mean','count'])

print("\nPivot (b):\n", pivot_b.round(3))

# ==============================
# (c) Title vs Pclass (Fare Median)
# ==============================

pivot_c = pd.pivot_table(df, values='Fare',
                         index='title', columns='Pclass',
                         aggfunc='median')

print("\nPivot (c):\n", pivot_c.round(2))

# Heatmap
plt.figure()
plt.imshow(pivot_c)
plt.xticks(range(len(pivot_c.columns)), pivot_c.columns)
plt.yticks(range(len(pivot_c.index)), pivot_c.index)

for i in range(len(pivot_c.index)):
    for j in range(len(pivot_c.columns)):
        plt.text(j, i, f"{pivot_c.iloc[i,j]:.1f}",
                 ha='center', va='center')

plt.title("Median Fare: Title vs Pclass")
plt.colorbar()
plt.show()