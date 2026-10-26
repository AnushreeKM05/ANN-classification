# Customer Churn Prediction Using Deep Learning

## 📌 Project Overview

This project predicts whether a bank customer is likely to leave the bank (churn) using a deep learning model built with TensorFlow and Keras.

The application provides an interactive web interface built with Streamlit, allowing users to enter customer details and receive a churn prediction along with the predicted churn probability.

## 🎯 Objectives

- Predict customer churn using a trained neural network.
- Perform preprocessing and feature encoding on customer data.
- Apply feature scaling before making predictions.
- Build an interactive web application using Streamlit.
- Understand how machine learning models can help businesses improve customer retention.

## 🛠️ Technologies Used

- **Python** — Programming language
- **TensorFlow / Keras** — Deep learning model
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Feature encoding and scaling
- **Streamlit** — Web application interface
- **Pickle** — Saving and loading preprocessing objects
- **Git & GitHub** — Version control and project hosting

## 📂 Project Structure

```text
dlproject1/
│
├── app.py
├── model.h5
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── deeplearning.ipynb
└── prediction.ipynb
```

## ⚙️ Features

- Interactive customer information form
- Geography and gender selection
- Customer age, credit score, balance, tenure, and salary inputs
- Credit card and active membership selection
- Automatic feature encoding and scaling
- Churn prediction using a trained deep learning model
- Display of predicted churn probability

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd dlproject1
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🧠 How It Works

1. **Data preprocessing:** Customer information is prepared for the model.
2. **Feature encoding:** Gender and geography are encoded using the saved preprocessing objects.
3. **Feature scaling:** Input values are transformed using the fitted scaler.
4. **Model prediction:** The trained neural network estimates the probability of customer churn.
5. **Result display:** The application displays the estimated probability and predicts whether the customer is likely to churn.

The application uses a probability threshold of `0.5` to classify customers:

- Probability greater than `0.5`: Customer is predicted to churn.
- Probability less than or equal to `0.5`: Customer is predicted not to churn.

## 📊 Model and Preprocessing Files

The application requires the following trained artifacts:

| File | Purpose |
|---|---|
| `model.h5` | Trained deep learning model |
| `label_encoder_gender.pkl` | Saved gender encoder |
| `onehot_encoder_geo.pkl` | Saved geography encoder |
| `scaler.pkl` | Saved feature scaler |

These files must correspond to the same training pipeline and feature order used to train the model.

## 🔮 Future Improvements

- Evaluate model performance using precision, recall, F1-score, and ROC-AUC.
- Add visualizations for customer churn insights.
- Improve the user interface and prediction explanations.
- Deploy the application to a cloud platform.
- Add model monitoring and data validation.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Predictions are estimates and should not be treated as definitive conclusions about individual customers.

## 👩‍💻 Author

**Anushree**

Developed as a deep learning project to explore customer churn prediction and interactive machine learning applications.
