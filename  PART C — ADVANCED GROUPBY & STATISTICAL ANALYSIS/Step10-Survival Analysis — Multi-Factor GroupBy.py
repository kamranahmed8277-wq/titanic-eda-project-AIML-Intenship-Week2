# Step 10 — Survival Analysis (Multi-Factor GroupBy)

# Objective:
# Analyze how different factors influence survival probability.

# ------------------------------------------------------------
# (a) Pclass
# ------------------------------------------------------------

# Observation:
# First class has highest survival (~0.63)
# Third class has lowest (~0.24)

# Insight:
# Higher socioeconomic status → better survival chances

# ------------------------------------------------------------
# (b) Sex
# ------------------------------------------------------------

# Observation:
# Female survival ≈ 0.74
# Male survival ≈ 0.19

# Insight:
# Strong evidence of "women first" evacuation policy

# ------------------------------------------------------------
# (c) Pclass + Sex
# ------------------------------------------------------------

# Observation:
# First-class females → extremely high survival
# Third-class males → extremely low survival

# Insight:
# Combined effect is very powerful

# ------------------------------------------------------------
# (d) Age Group
# ------------------------------------------------------------

# Observation:
# Children have relatively higher survival
# Adults dominate population

# Insight:
# Priority given to children

# ------------------------------------------------------------
# (e) Is Alone
# ------------------------------------------------------------

# Observation:
# Slight difference between alone vs family

# Insight:
# Weak predictor compared to others

# ------------------------------------------------------------
# (f) Embarked
# ------------------------------------------------------------

# Observation:
# Small variation between ports

# Insight:
# Weak influence

# ------------------------------------------------------------
# 🔥 STRONGEST PREDICTOR ANALYSIS
# ------------------------------------------------------------

# Sex is the STRONGEST predictor

# Evidence:
# Female survival ≈ 74%
# Male survival ≈ 19%

# Difference ≈ 55% (very large gap)

# Compare:
# Pclass difference ≈ 40%
# Age group difference ≈ smaller
# Others even smaller

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# Strongest predictor: Sex
# Second strongest: Pclass
# Combined effect (Pclass + Sex) gives best insights

# Survival was heavily influenced by:
# ✔ Gender
# ✔ Social class



import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (ensure correct path in PyCharm)
df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# BASIC PREPROCESSING
# ==============================

df['Age'] = df['Age'].fillna(
    df.groupby(['Pclass','Sex'])['Age'].transform('median')
)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['family_size'] = df['SibSp'] + df['Parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

# Age groups
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

# Overall survival rate
overall_rate = df['Survived'].mean()

# ==============================
# PLOT FUNCTION (NO COLORS SPECIFIED)
# ==============================

def plot_bar(data, title):
    ax = data.plot(kind='bar')

    plt.axhline(overall_rate)
    plt.title(title)
    plt.ylabel("Survival Rate")

    # Labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom')

    plt.show()

# ==============================
# (a) Pclass
# ==============================

pclass_sr = df.groupby('Pclass')['Survived'].mean().round(3)
print("\nPclass Survival:\n", pclass_sr)
plot_bar(pclass_sr, "Survival Rate by Pclass")

# ==============================
# (b) Sex
# ==============================

sex_sr = df.groupby('Sex')['Survived'].mean().round(3)
print("\nSex Survival:\n", sex_sr)
plot_bar(sex_sr, "Survival Rate by Sex")

# ==============================
# (c) Pclass + Sex
# ==============================

ps_sr = df.groupby(['Pclass','Sex'])['Survived'].mean().unstack().round(3)
print("\nPclass + Sex:\n", ps_sr)

ps_sr.plot(kind='bar')
plt.axhline(overall_rate)
plt.title("Survival Rate by Pclass & Sex")
plt.ylabel("Survival Rate")
plt.show()

# ==============================
# (d) Age Group
# ==============================

age_sr = df.groupby('age_group')['Survived'].mean().round(3)
print("\nAge Group Survival:\n", age_sr)
plot_bar(age_sr, "Survival Rate by Age Group")

# ==============================
# (e) Is Alone
# ==============================

alone_sr = df.groupby('is_alone')['Survived'].mean().round(3)
print("\nIs Alone Survival:\n", alone_sr)
plot_bar(alone_sr, "Survival Rate by Is Alone")

# ==============================
# (f) Embarked
# ==============================

emb_sr = df.groupby('Embarked')['Survived'].mean().round(3)
print("\nEmbarked Survival:\n", emb_sr)
plot_bar(emb_sr, "Survival Rate by Embarked")

print("\nOverall Survival Rate:", round(overall_rate,3))