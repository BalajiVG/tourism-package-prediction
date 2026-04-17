import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the trained model from the Hugging Face model hub and load it
model_path = hf_hub_download(
    repo_id="BalajiVG/tourism-package-model",
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# ---- Streamlit UI ----
st.title("Wellness Tourism Package Prediction App")
st.write(
    "This application predicts whether a customer is likely to purchase the newly introduced "
    "**Wellness Tourism Package**. Fill in the customer details below to get a prediction."
)

# ---- User input: Customer Details ----
st.header("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input(
    "Number of Persons Visiting", min_value=1, max_value=10, value=2
)
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input(
    "Number of Trips per year", min_value=0, max_value=25, value=2
)
passport = st.selectbox("Has Passport?", [0, 1])
own_car = st.selectbox("Owns a Car?", [0, 1])
number_of_children_visiting = st.number_input(
    "Number of Children (below 5) Visiting", min_value=0, max_value=5, value=0
)
designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
monthly_income = st.number_input(
    "Monthly Income", min_value=1000.0, max_value=200000.0, value=23000.0, step=100.0
)

# ---- User input: Customer Interaction Data ----
st.header("Customer Interaction Data")
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)
number_of_followups = st.number_input(
    "Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0, step=1.0
)
duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)", min_value=1.0, max_value=200.0, value=15.0, step=1.0
)

# ---- Assemble input into DataFrame (column order must match training) ----
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    result = "Will Purchase the Wellness Package" if prediction == 1 else "Will Not Purchase"
    st.subheader("Prediction Result")
    st.success(f"The model predicts: **{result}**")
    st.write(f"Probability of purchase: **{proba:.2%}**")
