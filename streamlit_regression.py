
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Page configuration
st.set_page_config(
    page_title="Estimated Salary Prediction",
    page_icon="💰"
)

# Load trained model
model = tf.keras.models.load_model("regression_model.h5")

# Load encoders and scaler
with open("onehotencoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scalar.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)


# Streamlit app
st.title("Estimated Salary Prediction")

st.write("Enter customer details to predict estimated salary.")

# User inputs
geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider("Age", 18, 92)

balance = st.number_input("Balance", min_value=0.0)

credit_score = st.number_input(
    "Credit Score", min_value=300, max_value=900, value=600
)

exited = st.selectbox("Exited", [0, 1])

tenure = st.slider("Tenure", 0, 10)

num_of_products = st.slider("Number of Products", 1, 4)

has_cr_card = st.selectbox("Has Credit Card", [0, 1])

is_active_member = st.selectbox("Is Active Member", [0, 1])


# Prediction
if st.button("Predict Estimated Salary"):

    # Gender encoding
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Input dataframe
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "Exited": [exited]
    })

    # Geography encoding
    geo_encoded = onehot_encoder_geo.transform(
        pd.DataFrame({"Geography": [geography]})
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # Combine features
    input_data = pd.concat(
        [input_data, geo_encoded_df],
        axis=1
    )

    # Match training feature order if available
    if hasattr(scaler, "feature_names_in_"):
        input_data = input_data.loc[:, scaler.feature_names_in_]

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled, verbose=0)

    # Display result
    st.success("Prediction completed!")

    st.metric(
        "Estimated Salary",
        f"${float(np.asarray(prediction).flatten()[0]):,.2f}"
    )
