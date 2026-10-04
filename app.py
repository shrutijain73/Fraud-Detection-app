import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")

# ---------- Project description ----------
st.title("Fraud Detection Prediction App")
st.markdown(
    """
This app uses a machine learning model trained on transaction data to estimate
whether a transaction is **fraudulent**. Enter the transaction details below and
click **Predict**.

**How it works:** the model looks at the transaction type, the amount, and the
sender and receiver balances before and after the transaction.
"""
)

st.divider()

# ---------- Inputs ----------
transaction_type = st.selectbox(
    "Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
)
amount = st.number_input("Amount", min_value=0.0, value=1000.0)

oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=1000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=1000.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=1000.0)

# ---------- Input checks (warnings only, they don't block the prediction) ----------
tolerance = 0.01

if abs((oldbalanceOrg - amount) - newbalanceOrig) > tolerance:
    st.warning(
        "Sender balance doesn't add up: old balance - amount should equal "
        "the new balance."
    )

if transaction_type != "PAYMENT" and abs((oldbalanceDest + amount) - newbalanceDest) > tolerance:
    st.warning(
        "Receiver balance doesn't add up: old balance + amount should equal "
        "the new balance."
    )

if amount == 0:
    st.warning("Amount is 0. Please enter a transaction amount.")

# ---------- Prediction ----------
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }])

    prediction = model.predict(input_data)[0]
    fraud_probability = model.predict_proba(input_data)[0][1]  # probability of class 1 (fraud)

    st.subheader(f"Prediction: {int(prediction)}")
    st.metric("Fraud probability", f"{fraud_probability * 100:.2f}%")
    st.progress(float(fraud_probability))

    if prediction == 1:
        st.error("This transaction can be fraud")
    else:
        st.success("This transaction looks like it is not a fraud")