import streamlit as st
import pandas as pd
import joblib

# 1. Load the "brain"
model = joblib.load("fraud_model_logistic.pkl")

# 2. UI Setup
st.title("🕵️‍♂️ Fraud Detection System")
st.markdown("Enter transaction details below to check for suspicious activity.")

# 3. User Inputs
col1, col2 = st.columns(2)

with col1:
    ttype = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    amount = st.number_input("Amount ($)", min_value=0.0, value=1000.0)
    old_org = st.number_input("Sender Initial Balance", min_value=0.0, value=5000.0)
    new_org = st.number_input("Sender Final Balance", min_value=0.0, value=4000.0)

with col2:
    old_dest = st.number_input("Receiver Initial Balance", min_value=0.0, value=0.0)
    new_dest = st.number_input("Receiver Final Balance", min_value=0.0, value=1000.0)

# 4. Logic & Prediction
if st.button("Run Fraud Check"):
    # Create the engineered features your model needs
    diff_org = old_org - new_org
    diff_dest = old_dest - new_dest
    
    # Put data into a DataFrame (must match column names in p1.py)
    input_df = pd.DataFrame([{
        'type': ttype,
        'amount': amount,
        'oldbalanceOrg': old_org,
        'newbalanceOrig': new_org,
        'oldbalanceDest': old_dest,
        'newbalanceDest': new_dest,
        'diff_Org': diff_org,
        'diff_Dest': diff_dest
    }])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display Result
    if prediction == 1:
        st.error("🚨 ALERT: This transaction is flagged as POTENTIAL FRAUD.")
    else:
        st.success("✅ CLEAR: This transaction looks normal.")