
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')
st.write('Enter customer details below.')

# User input
geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.number_input(
    'Age', min_value=18, max_value=92, value=30, step=1
)

balance = st.number_input(
    'Balance', min_value=0.0, value=0.0, step=100.0
)

credit_score = st.number_input(
    'Credit Score', min_value=0, max_value=1000,
    value=650, step=1
)

estimated_salary = st.number_input(
    'Estimated Salary', min_value=0.0,
    value=50000.0, step=1000.0
)

tenure = st.number_input(
    'Tenure', min_value=0, max_value=10, value=5, step=1
)

num_of_products = st.number_input(
    'Number of Products', min_value=1, max_value=4,
    value=1, step=1
)

has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender_0': [
        1 if gender == label_encoder_gender.classes_[0] else 0
    ],
    'Gender_1': [
        1 if gender == label_encoder_gender.classes_[1] else 0
    ],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine encoded Geography with input data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Predict churn
if st.button('Predict Churn'):
    try:
        expected_columns = list(scaler.feature_names_in_)

        missing_columns = [
            col for col in expected_columns
            if col not in input_data.columns
        ]

        if missing_columns:
            st.error(f'Missing required features: {missing_columns}')
        else:
            input_data = input_data[expected_columns]
            input_data_scaled = scaler.transform(input_data)

            prediction = model.predict(
                input_data_scaled, verbose=0
            )
            probability = float(prediction[0][0])

            st.metric(
                'Churn Probability',
                f'{probability:.2%}'
            )

            if probability > 0.5:
                st.error('The customer is likely to churn.')
            else:
                st.success('The customer is unlikely to churn.')

    except Exception as e:
        st.error(f'Prediction error: {e}')
