# Step 15 — Professional Titanic Dashboard

# Objective:
# Create a 6-chart dashboard summarizing key insights

# ------------------------------------------------------------
# Chart 1 — Survival by Pclass
# ------------------------------------------------------------

# Insight:
# First-class passengers have highest survival

# ------------------------------------------------------------
# Chart 2 — Age Distribution
# ------------------------------------------------------------

# Insight:
# Younger passengers slightly more likely to survive

# ------------------------------------------------------------
# Chart 3 — Fare Distribution
# ------------------------------------------------------------

# Insight:
# Higher fares strongly linked to higher class

# ------------------------------------------------------------
# Chart 4 — Heatmap (Pclass × Sex)
# ------------------------------------------------------------

# Insight:
# Females in first class have highest survival

# ------------------------------------------------------------
# Chart 5 — Family Size
# ------------------------------------------------------------

# Insight:
# Medium family size performs best

# ------------------------------------------------------------
# Chart 6 — Title
# ------------------------------------------------------------

# Insight:
# Social titles reflect survival differences

# ------------------------------------------------------------
# FINAL CONCLUSION
# ------------------------------------------------------------

# Survival depends on:
# ✔ Gender (strongest)
# ✔ Class
# ✔ Fare
# ✔ Family structure

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(r"C:\Users\J J LAPTOP\Desktop\ DATA ANALYSIS WITH NUMPY &  PANDAS\Dataset Titanic — Machine Learning from Disaster\train.csv")

# ==============================
# BASIC FEATURE ENGINEERING (FIX MISSING COLUMN ISSUE)
# ==============================

# Family Size (FIX FOR YOUR ERROR)
df['family_size'] = df['SibSp'] + df['Parch'] + 1

# Is Alone (optional but useful)
df['is_alone'] = (df['family_size'] == 1).astype(int)

# Title feature
df['title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
df['title'] = df['title'].apply(lambda x: x if x in common_titles else 'Rare')

# ==============================
# CREATE FIGURE
# ==============================

fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# ==============================
# (1) Survival Rate by Pclass
# ==============================

survival_pclass = df.groupby('Pclass')['Survived'].mean()

axes[0, 0].bar(survival_pclass.index, survival_pclass.values)
axes[0, 0].set_title("Survival Rate by Pclass")
axes[0, 0].set_xlabel("Pclass")
axes[0, 0].set_ylabel("Survival Rate")

for i, v in enumerate(survival_pclass.values):
    axes[0, 0].text(i+1, v, f"{v:.2f}", ha='center')

# ==============================
# (2) Age Distribution (KDE)
# ==============================

sns.kdeplot(df[df['Survived']==1]['Age'], ax=axes[0,1], label='Survived', fill=True)
sns.kdeplot(df[df['Survived']==0]['Age'], ax=axes[0,1], label='Not Survived', fill=True)

axes[0,1].set_title("Age Distribution by Survival")
axes[0,1].legend()

# ==============================
# (3) Fare Boxplot by Pclass
# ==============================

sns.boxplot(x='Pclass', y='Fare', data=df, ax=axes[1,0])
axes[1,0].set_yscale('log')
axes[1,0].set_title("Fare Distribution by Pclass (Log Scale)")

# ==============================
# (4) Heatmap: Pclass × Sex
# ==============================

pivot = pd.pivot_table(df, values='Survived',
                       index='Pclass', columns='Sex',
                       aggfunc='mean')

sns.heatmap(pivot, annot=True, fmt='.2f', ax=axes[1,1])
axes[1,1].set_title("Survival Heatmap (Pclass × Sex)")

# ==============================
# (5) Family Size vs Survival (FIXED)
# ==============================

family_survival = df.groupby('family_size')['Survived'].mean()

axes[2,0].plot(family_survival.index, family_survival.values, marker='o')

axes[2,0].set_title("Family Size vs Survival Rate")
axes[2,0].set_xlabel("Family Size")
axes[2,0].set_ylabel("Survival Rate")

# ==============================
# (6) Survival by Title
# ==============================

title_survival = pd.crosstab(df['title'], df['Survived'], normalize='index')

title_survival.plot(kind='bar', stacked=True, ax=axes[2,1])

axes[2,1].set_title("Survival Proportion by Title")
axes[2,1].set_ylabel("Proportion")

# ==============================
# FINAL TOUCH
# ==============================

plt.suptitle("Titanic EDA Dashboard — Fixed Version", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("titanic_dashboard.png", dpi=150)
plt.show()