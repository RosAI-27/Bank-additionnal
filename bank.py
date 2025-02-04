import streamlit as st
import joblib
import pandas as pd


# Load the trained model and scaler
scaler = joblib.load('scaler.pkl')
# Load the model
model = joblib.load('logistic_regression_model.pkl')

# Streamlit app
st.title("Bank Marketing Campaign Prediction")

# Input fields for numerical features
st.sidebar.header("Client Information")
age = st.sidebar.number_input("Age", min_value=18, max_value=100)
duration = st.sidebar.number_input("Duration (last contact duration in seconds)", min_value=0)


# Input fields for categorical features
job = st.sidebar.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.sidebar.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
education = st.sidebar.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"])
default = st.sidebar.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
housing = st.sidebar.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
loan = st.sidebar.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])

# Predict button
if st.sidebar.button("Predict Subscription"):
    # Create a DataFrame with all input features
    input_data = pd.DataFrame({
        'age': [age],
        'duration': [duration],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
      
    })

    # One-hot encode categorical features (if your model expects encoded data)
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Scale the input data (if your model expects scaled data)
    input_scaled = scaler.transform(input_data)
     # Make prediction
    prediction = model.predict(input_scaled)[0]


    # Show result
    st.success("Prediction: " + ("Yes" if prediction == 1 else "No"))
