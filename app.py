# ─────────────────────────────────────────────
# CUSTOMER CHURN PREDICTION — STREAMLIT UI
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔍",
    layout="centered"
)

# ─────────────────────────────────────────────
# TRAIN MODEL (runs once in background)
# ─────────────────────────────────────────────

@st.cache_resource
def train_model():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    le = LabelEncoder()
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = le.fit_transform(df[col])

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    return model, scaler, X.columns.tolist()

model, scaler, feature_cols = train_model()

# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────

st.title("🔍 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict if they will churn.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

with col2:
    st.subheader("📱 Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

col3, col4 = st.columns(2)

with col3:
    st.subheader("💳 Account Info")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col4:
    st.subheader("💰 Charges")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges ($)", 18, 120, 65)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly))

st.divider()

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

if st.button("🔍 PREDICT CHURN", use_container_width=True, type="primary"):

    # Build input row
    input_data = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone == 'Yes' else 0,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'PaperlessBilling': 1 if paperless == 'Yes' else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No internet service': 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 0,
        'DeviceProtection_Yes': 0,
        'TechSupport_No internet service': 0,
        'TechSupport_Yes': 0,
        'StreamingTV_No internet service': 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 0,
        'StreamingMovies_Yes': 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
    }

    # Create dataframe with correct columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ This customer is LIKELY TO CHURN!")
        st.metric("Churn Probability", f"{probability*100:.1f}%")
        st.markdown("**💡 Suggestion:** Offer a discount or upgrade to a long-term contract.")
    else:
        st.success(f"✅ This customer is LIKELY TO STAY!")
        st.metric("Churn Probability", f"{probability*100:.1f}%")
        st.markdown("**💡 Great!** This customer appears satisfied with the service.")

    # Probability bar
    st.progress(float(probability))
