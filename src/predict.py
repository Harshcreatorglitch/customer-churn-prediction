import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/customer_churn_model.joblib")


# Example customer
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0
}


# Convert customer information into DataFrame
customer_df = pd.DataFrame([customer])


# Make prediction
prediction = model.predict(customer_df)[0]

# Get probability of churn
churn_probability = model.predict_proba(customer_df)[0][1]


if prediction == 1:
    result = "Likely to Churn"
else:
    result = "Likely to Stay"


print("Prediction:", result)
print(f"Churn Probability: {churn_probability:.2%}")