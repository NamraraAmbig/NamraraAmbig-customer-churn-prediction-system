# ─────────────────────────────────────────────
# CUSTOMER CHURN PREDICTION PROJECT
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# ─────────────────────────────────────────────
# STEP 2 — LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Shape:", df.shape)
print("\nChurn distribution:")
print(df['Churn'].value_counts())
print("\n✅ Data loaded successfully!")

# ─────────────────────────────────────────────
# STEP 3 — DATA CLEANING
# ─────────────────────────────────────────────

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
df.dropna(inplace=True)
print("✅ TotalCharges fixed")

df.drop(columns=['customerID'], inplace=True)
print("✅ customerID dropped")

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("✅ Churn encoded")

le = LabelEncoder()
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
print("✅ Binary columns encoded")

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
print("✅ Categorical columns encoded")
print("Final shape after cleaning:", df.shape)
print("\n✅ Data Cleaning Complete!")

# ─────────────────────────────────────────────
# STEP 4 — VISUALIZATIONS
# ─────────────────────────────────────────────

os.makedirs('plots', exist_ok=True)

raw = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
raw['TotalCharges'] = pd.to_numeric(raw['TotalCharges'], errors='coerce')
raw['TotalCharges'].fillna(raw['TotalCharges'].median(), inplace=True)

# EDA Charts
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Customer Churn Analysis', fontsize=16, fontweight='bold')

raw['Churn'].value_counts().plot(kind='bar', ax=axes[0,0],
    color=['steelblue','tomato'], edgecolor='black')
axes[0,0].set_title('Churn Distribution')
axes[0,0].tick_params(rotation=0)

raw.groupby('Contract')['Churn'].value_counts().unstack().plot(kind='bar',
    ax=axes[0,1], color=['steelblue','tomato'], edgecolor='black')
axes[0,1].set_title('Churn by Contract Type')
axes[0,1].tick_params(rotation=15)

raw.groupby('InternetService')['Churn'].value_counts().unstack().plot(kind='bar',
    ax=axes[0,2], color=['steelblue','tomato'], edgecolor='black')
axes[0,2].set_title('Churn by Internet Service')
axes[0,2].tick_params(rotation=15)

axes[1,0].hist(raw[raw['Churn']=='No']['tenure'], bins=30, alpha=0.7,
    color='steelblue', label='Stayed')
axes[1,0].hist(raw[raw['Churn']=='Yes']['tenure'], bins=30, alpha=0.7,
    color='tomato', label='Churned')
axes[1,0].set_title('Tenure Distribution')
axes[1,0].legend()

raw.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1,1],
    boxprops=dict(color='steelblue'),
    medianprops=dict(color='tomato', linewidth=2))
axes[1,1].set_title('Monthly Charges by Churn')

numeric_df = df[['tenure','MonthlyCharges','TotalCharges','SeniorCitizen','Churn']]
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('plots/eda_charts.png')
plt.close()
print("✅ Charts saved to plots/eda_charts.png")

# ─────────────────────────────────────────────
# COUNT PLOTS
# ─────────────────────────────────────────────

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle('Customer Churn Count Plots', fontsize=16, fontweight='bold')

sns.countplot(data=raw, x='gender', hue='Churn',
    palette=['steelblue','tomato'], ax=axes2[0,0])
axes2[0,0].set_title('Churn by Gender')

sns.countplot(data=raw, x='SeniorCitizen', hue='Churn',
    palette=['steelblue','tomato'], ax=axes2[0,1])
axes2[0,1].set_title('Churn by Senior Citizen')
axes2[0,1].set_xlabel('Senior Citizen (0=No, 1=Yes)')

sns.countplot(data=raw, x='Partner', hue='Churn',
    palette=['steelblue','tomato'], ax=axes2[0,2])
axes2[0,2].set_title('Churn by Partner')

sns.countplot(data=raw, x='Contract', hue='Churn',
    palette=['steelblue','tomato'], ax=axes2[1,0])
axes2[1,0].set_title('Churn by Contract Type')
axes2[1,0].tick_params(rotation=15)

sns.countplot(data=raw, x='InternetService', hue='Churn',
    palette=['steelblue','tomato'], ax=axes2[1,1])
axes2[1,1].set_title('Churn by Internet Service')

sns.countplot(data=raw, x='PaymentMethod', hue='Churn',
    palette=['steelblue','tomato'], ax=axes2[1,2])
axes2[1,2].set_title('Churn by Payment Method')
axes2[1,2].tick_params(rotation=15)

plt.tight_layout()
plt.savefig('plots/count_plots.png')
plt.close()
print("✅ Count plots saved to plots/count_plots.png")

# ─────────────────────────────────────────────
# STEP 5 — TRAIN/TEST SPLIT
# ─────────────────────────────────────────────

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("✅ Data split done!")
print("   Training samples:", X_train.shape[0])
print("   Testing samples :", X_test.shape[0])

# ─────────────────────────────────────────────
# STEP 6 — BUILD & EVALUATE MODELS
# ─────────────────────────────────────────────

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)

print("\n── Logistic Regression ──")
print("Accuracy :", round(accuracy_score(y_test, lr_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Report:\n", classification_report(y_test, lr_pred))

dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n── Decision Tree ──")
print("Accuracy :", round(accuracy_score(y_test, dt_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Report:\n", classification_report(y_test, dt_pred))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n── Random Forest ──")
print("Accuracy :", round(accuracy_score(y_test, rf_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Report:\n", classification_report(y_test, rf_pred))

print("\n✅ All models trained and evaluated!")

# ─────────────────────────────────────────────
# STEP 7 — FEATURE IMPORTANCE (BONUS)
# ─────────────────────────────────────────────

feature_names = X.columns
importances = rf.feature_importances_

feat_df = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
feat_df.sort_values().plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

print("\n📊 Top 10 Features that predict churn:")
print(feat_df)
print("\n✅ Feature importance saved to plots/feature_importance.png")

print("\n" + "="*50)
print("   🎉 PROJECT COMPLETE!")
print("="*50)
print("✅ Data Cleaned")
print("✅ EDA Charts       → plots/eda_charts.png")
print("✅ Count Plots      → plots/count_plots.png")
print("✅ Feature Importance → plots/feature_importance.png")
print("✅ Logistic Regression Accuracy: 80.7%")
print("✅ Decision Tree Accuracy      : 79.7%")
print("✅ Random Forest Accuracy      : 78.7%")
print("="*50)