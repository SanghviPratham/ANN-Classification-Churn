import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
import streamlit as st

# Load model
model = tf.keras.models.load_model('model.h5')

# Load encoders & scaler
with open('onhot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction")

input_data = {
    'CreditScore': st.number_input("Credit Score", 300, 850, 600),
    'Geography': st.selectbox("Geography", onehot_encoder_geo.categories_[0]),
    'Gender': st.selectbox("Gender", label_encoder_gender.classes_),
    'Age': st.number_input("Age", 18, 100, 30),
    'Tenure': st.number_input("Tenure", 0, 10, 3),
    'Balance': st.number_input("Balance", 0.0, value=10000.0),
    'NumOfProducts': st.number_input("Number of Products", 1, 4, 1),
    'HasCrCard': st.selectbox("Has Credit Card", [0, 1]),
    'IsActiveMember': st.selectbox("Is Active Member", [0, 1]),
    'EstimatedSalary': st.number_input("Estimated Salary", 0.0, value=50000.0)
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ✅ Encode Geography (OneHotEncoder)
geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# ✅ Encode Gender (LabelEncoder)
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# Drop original Geography and add encoded columns
input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

# ✅ Ensure same column order as training
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Output
if prediction_proba >= 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")