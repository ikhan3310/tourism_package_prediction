import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="ink85/tourism-package-prediction", filename="tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Travel Package Purchase Prediction")
st.markdown("Enter customer details to predict if they will take the product (**ProdTaken**).")

# --- SECTION 1: Customer Demographics ---
st.subheader("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"]) # Note: specific to this dataset's known typos, or just Male/Female
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=20000.0, step=500.0)

with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    # Converting Yes/No to 1/0 for the model
    passport_input = st.selectbox("Has Passport?", ["Yes", "No"])
    passport = 1 if passport_input == "Yes" else 0

    own_car_input = st.selectbox("Owns Car?", ["Yes", "No"])
    own_car = 1 if own_car_input == "Yes" else 0

# --- SECTION 2: Trip & Interaction Data ---
st.markdown("---")
st.subheader("Interaction & Trip Data")
col3, col4 = st.columns(2)

with col3:
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score (1-5)", [1, 2, 3, 4, 5], index=2)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=120.0, value=10.0)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)

with col4:
    preferred_property_star = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
    number_of_trips = st.number_input("Avg Number of Trips (Annually)", min_value=0, max_value=50, value=2)
    number_of_person = st.number_input("Total Persons Visiting", min_value=1, max_value=20, value=2)
    number_of_children = st.number_input("Children Visiting (< 5y)", min_value=0, max_value=10, value=0)

# --- Assemble input into DataFrame ---
# Note: Ensure these keys match exactly what your model was trained on
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch
}])

# --- Predict button ---
st.markdown("---")
if st.button("Predict Purchase"):
    # Assuming 'model' is already loaded in your environment
    # prediction = model.predict(input_data)[0]

    # Placeholder logic for demonstration (Replace with actual model.predict above)
    st.info("⚠️ This is a UI demo. Connect your trained model to generate real predictions.")
    # Example simulation:
    prediction = 1 # Simulated result

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("Result: **Package Purchased** (ProdTaken = 1)")
    else:
        st.warning("Result: **No Purchase** (ProdTaken = 0)")

    # Optional: View the raw input data being sent to model
    with st.expander("View Input Data"):
        st.dataframe(input_data)
