import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load Model
model = tf.keras.models.load_model('model.h5')

# Load encoders & scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title
st.title('Customer Churn Prediction')

# Input fields
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.number_input('Credit Score', min_value=0)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 30)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create base input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode Gender (Label Encoder)
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# One-hot encode Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine numerical + encoded categorical features
input_final = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale final input
input_data_scaled = scaler.transform(input_final)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]  # Assuming output is sigmoid

# Output
if prediction_proba >= 0.5:
    st.success(f'Customer is likely to CHURN ({prediction_proba:.2f})')
else:
    st.info(f'Customer is NOT likely to churn ({prediction_proba:.2f})')
