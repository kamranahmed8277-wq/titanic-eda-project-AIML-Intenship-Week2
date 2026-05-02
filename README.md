📄 Titanic EDA Project
📌 Overview

This project performs Exploratory Data Analysis (EDA) on the Titanic dataset to understand the key factors that influenced passenger survival. It includes data cleaning, feature engineering, statistical analysis, and visualization.

🎯 Objective

The main goal is to:

Analyze survival patterns
Identify important features affecting survival
Prepare dataset for machine learning models

📊 Dataset
Source: Titanic Dataset (Kaggle)
Features include:
Passenger details (Age, Sex, Class, Fare)
Family information (SibSp, Parch)
Survival status

🧹 Data Cleaning
Missing Age values filled using median grouped by Pclass & Sex
Missing Embarked values filled using mode
Cabin column processed into:
has_cabin
deck feature


⚙️ Feature Engineering

New features created:

family_size → SibSp + Parch + 1
is_alone → whether passenger traveled alone
age_group → Child / Teen / Adult / Senior
title → extracted from name (Mr, Mrs, etc.)
fare_per_person → normalized fare per passenger


📊 Key Insights
Females had significantly higher survival rates than males
First-class passengers had the highest survival probability
Children had better survival chances than adults
Small family size improved survival chances


📈 Visualizations
Survival rate by class, gender, age group
Heatmaps for class vs sex survival
KDE plots for age distribution
Boxplots for fare distribution
Family size vs survival trends


🧠 Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn

🚀 How to Run
git clone https://github.com/kamranahmed8277-wq/titanic-eda-project-AIML-Intenship-Week2.git
cd titanic-eda-project
python titanic_eda_project.py

📌 Future Improvements
Machine learning model (Logistic Regression / Random Forest)
Feature selection optimization
Streamlit dashboard for interactive analysis

👤 Author

Kamran Ahmed
Data Analysis | Python | Machine Learning Learner
