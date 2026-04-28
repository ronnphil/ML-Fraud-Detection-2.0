# 🕵️‍♂️ Financial Fraud Detection System 2.0

An end-to-end Machine Learning pipeline designed to identify fraudulent transactions in real-time. This project handles a massive dataset of 6.3M+ transactions and provides a user-friendly Streamlit interface for instant predictions.

## 📊 Performance Summary
| Metric | Value |
| :--- | :--- |
| **Recall** | **94%** (Caught 94% of all fraud cases) |
| **Accuracy** | **95%** (Overall correct prediction rate) |
| **Model** | Logistic Regression (Balanced Class Weights) |

## 🚀 Key Features
* **Real-Time Prediction:** Integrated with a **Streamlit** web app.
* **Big Data Handling:** Processed 6.3M rows using Pandas.
* **Balanced Detection:** Optimized for high recall to ensure security-first results.

## 🏃‍♂️ Setup & Usage
1. **Install Requirements:**
   ```bash
   pip install pandas scikit-learn streamlit joblib