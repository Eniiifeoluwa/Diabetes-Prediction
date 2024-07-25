import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import warnings
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.models import Sequential, Model
from tensorflow.compat.v1.keras.layers import Input, Dense
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import keras

import shap
from sklearn.metrics import roc_curve, roc_auc_score
from keras.models import load_model
data = pd.read_csv("Transformed Data.csv")
# Separating features and target variable
X = data.drop('diabetes', axis=1)
Y = data['diabetes']

# Standardize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Use SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)


# Load the trained model
model = load_model("diabetes_model.h5")


# Define feature columns
feature_columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                   'smoking_history_never smoked', 'smoking_history_smoked before', 'smoking_history_smoking',
                   'gender_female', 'gender_male']

# Creating a simple Streamlit app
st.title("Diabetes Prediction App")

# Get user input
age = st.number_input("Age", min_value=0, max_value=120, value=50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=28.7)
hba1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=15.0, value=6.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0, max_value=300, value=150)
smoking_history = st.selectbox("Smoking History", ["never smoked", "smoked before", "smoking"])
gender = st.selectbox("Gender", ["male", "female"])

# Preprocess the input data
smoking_history_never_smoked = 1 if smoking_history == "never smoked" else 0
smoking_history_smoked_before = 1 if smoking_history == "smoked before" else 0
smoking_history_smoking = 1 if smoking_history == "smoking" else 0
gender_female = 1 if gender == "female" else 0
gender_male = 1 if gender == "male" else 0

new_data = pd.DataFrame({
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'bmi': [bmi],
    'HbA1c_level': [hba1c_level],
    'blood_glucose_level': [blood_glucose_level],
    'smoking_history_never smoked': [smoking_history_never_smoked],
    'smoking_history_smoked before': [smoking_history_smoked_before],
    'smoking_history_smoking': [smoking_history_smoking],
    'gender_female': [gender_female],
    'gender_male': [gender_male],
})

# Normalize the data using MinMaxScaler (fit on training data)
# Assuming you have saved the scaler used during training
scaler = MinMaxScaler() 
scaler.fit(X)
new_data_scaled = scaler.transform(new_data)

# Make predictions
# Add a button to trigger prediction
if st.button("Predict"):
    # Make predictions
    predictions = model.predict(new_data_scaled)
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Display the prediction result
    if binary_predictions[0][0] == 1:
        st.write("## Predicted class: **Diabetic**")
        st.write("### Predicted probability:", predictions[0][0])
        st.write("### Recommendations:")
        st.write("- Consult a healthcare provider for further diagnosis and treatment.")
        st.write("- Maintain a healthy diet and exercise regularly.")
        st.write("- Monitor your blood glucose levels frequently.")
    else:
        st.write("## Predicted class: **Not Diabetic**")
        st.write("### Predicted probability:", predictions[0][0])
        st.write("### Recommendations:")
        st.write("- Maintain a healthy lifestyle to prevent the onset of diabetes.")
        st.write("- Regular health check-ups are recommended.")
        st.write("- Continue monitoring your health indicators.")

    # Add more detailed explanations if needed
    st.write("### Explanation:")
    st.write("The model predicts the probability of diabetes based on your input features. "
             "A probability greater than 0.5 indicates a higher likelihood of diabetes.")
