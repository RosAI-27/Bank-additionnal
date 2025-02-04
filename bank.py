import streamlit as st
import joblib
import pandas as pd
import gdown

# Download the model file from Google Drive
file_id = "11lJWYbmvcxdU1gyfDTuw4VT2n_z5qjvV"
url = f"https://drive.google.com/uc?id={file_id}"
output = "bank_model.pkl"
gdown.download(url, output, quiet=False)

# Load the model
model = joblib.load(output)

# Load the scaler (if needed)
scaler = joblib.load('scaler.pkl')  # Ensure scaler.pkl is in your repo

# Streamlit app
st.title("Bank Marketing Campaign Prediction")

# Input fields for numerical features
st.sidebar.header("Client Information")
age = st.sidebar.number_input("Age", min_value=18, max_value=100)
duration = st.sidebar.number_input("Duration (last contact duration in seconds)", min_value=0)
campaign = st.sidebar.number_input("Number of contacts during this campaign", min_value=1)
pdays = st.sidebar.number_input("Days since last contact (999 if not contacted)", min_value=0, max_value=999)
previous = st.sidebar.number_input("Number of contacts before this campaign", min_value=0)
emp_var_rate = st.sidebar.number_input("Employment Variation Rate", value=0.0)
cons_price_idx = st.sidebar.number_input("Consumer Price Index", value=0.0)
cons_conf_idx = st.sidebar.number_input("Consumer Confidence Index", value=0.0)
euribor3m = st.sidebar.number_input("Euribor 3 Month Rate", value=0.0)
nr_employed = st.sidebar.number_input("Number of Employees", value=0.0)

# Input fields for categorical features
job = st.sidebar.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.sidebar.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
education = st.sidebar.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"])
default = st.sidebar.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
housing = st.sidebar.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
loan = st.sidebar.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
contact = st.sidebar.selectbox("Contact Communication Type", ["cellular", "telephone"])
month = st.sidebar.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
day_of_week = st.sidebar.selectbox("Last Contact Day of the Week", ["mon", "tue", "wed", "thu", "fri"])
poutcome = st.sidebar.selectbox("Outcome of Previous Marketing Campaign", ["failure", "nonexistent", "success"])

# Predict button
if st.sidebar.button("Predict Subscription"):
    # Create a DataFrame with all input features
    input_data = pd.DataFrame({
        'age': [age],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'poutcome': [poutcome]
    })

    # One-hot encode categorical features (if your model expects encoded data)
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Scale the input data (if your model expects scaled data)
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Show result
    st.success("Prediction: " + ("Yes" if prediction == 1 else "No"))
