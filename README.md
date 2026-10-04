# Fraud Detection Using Machine Learning

A machine learning web app that predicts whether a financial transaction is fraudulent.

**Live demo:** - https://shruti-fraud-detection.streamlit.app/

## About the project
Fraud is rare but costly. This project trains a classification model on transaction data
and serves it through a Streamlit app. The user enters transaction details and gets a
prediction along with the fraud probability.

## Dataset
- Source: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download
- Size: 6362620 rows
- Features: transaction type, amount, sender and receiver balances before and after the transaction , source account , destination account , 
- Target: fraud (1) or not fraud (0)


## Approach
1. Exploratory data analysis (EDA)
2. Feature engineering
3. Model training: Logistic Regression
4. Evaluation using precision, recall, F1-score and confusion matrix
5. Deployment with Streamlit

## Author
Shruti Jain 
