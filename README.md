# 🔍 Customer Churn Prediction

A Machine Learning project to predict customer churn 
using Telco Customer dataset from Kaggle.

## 📊 Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 80.7% |
| Decision Tree | 79.7% |
| Random Forest | 78.7% |

## 🔑 Key Insights
- TotalCharges, Tenure and MonthlyCharges are top predictors
- Month-to-month contract customers churn the most
- Electronic check payment users churn more
- Senior citizens have higher churn rate

## 🛠️ Tech Stack
Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit

## 📁 Project Structure
churn_prediction.py — Main ML code
app.py — Streamlit UI
plots/ — All charts and visualizations

## ▶️ How to Run
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
py churn_prediction.py
py -m streamlit run app.py
