# ANN-Churn-Model
📘 Customer Churn Prediction using Artificial Neural Networks (ANN)

This project predicts whether a customer will churn leave the bank using a trained Artificial Neural Network built with TensorFlow/Keras, preprocessed using scikit-learn, and deployed using Streamlit.

The model is trained on a typical Bank Churn Dataset and uses demographic & account details like credit score, age, geography, balance, etc., to estimate churn probability.

🚀 Project Features

🧠 Artificial Neural Network (ANN) for binary classification

✨ One-Hot Encoding for categorical features

🔢 Scaling using StandardScaler

📦 Saved model (model.h5)

🎛️ Encoders + Scaler saved as pickle files

🌐 Deployed frontend using Streamlit

📊 Outputs probability of churn and final prediction

▶️ How to Run the App Locally
1. Clone the repository

2. Create virtual environment
python -m venv venv
venv\Scripts\activate     # For Windows

3. Install dependencies
pip install -r requirements.txt

4. Run Streamlit app
streamlit run app.py
