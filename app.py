import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🧠")

# --- LOAD ASSETS ---
# Using absolute paths to prevent "FileNotFound" errors
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'ann_model.h5')
scaler_path = os.path.join(base_path, 'scaler.joblib')

@st.cache_resource
def load_assets():
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model = load_model(model_path,compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

ann, sc = load_assets()

# --- UI INTERFACE ---
st.title("🧠 ANN Customer Churn Analysis")
st.write("Determine if a customer will **Stay** or **Leave** based on their profile.")

if ann is None:
    st.error("⚠️ Error: 'ann_model.h5' or 'scaler.joblib' not found in the directory!")
    st.stop()

# --- INPUT SECTION ---
st.subheader("Customer Demographics")
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 40)
    credit_score = st.number_input("Credit Score", value=600)
    tenure = st.number_input("Tenure (Years)", value=3)

with col2:
    balance = st.number_input("Balance ($)", value=60000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    has_crcard = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active = st.radio("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary ($)", value=50000.0)

# --- PREDICTION LOGIC ---
if st.button("Analyze Customer"):
    # 1. Encoding Geography (Drop First: France = 0,0)
    # The order must be: Germany, Spain
    if geography == "Germany":
        geo_encoded = [1, 0]
    elif geography == "Spain":
        geo_encoded = [0, 1]
    else:  # France
        geo_encoded = [0, 0]

    # 2. Binary Mappings
    gender_val = 1 if gender == "Male" else 0
    card_val = 1 if has_crcard == "Yes" else 0
    active_val = 1 if is_active == "Yes" else 0

    # 3. Create Feature Array (Must be 11 columns)
    # Order: Germany, Spain, CreditScore, Gender, Age, Tenure, Balance, Products, HasCard, Active, Salary
    raw_features = np.array([geo_encoded + [
        credit_score, gender_val, age, tenure, 
        balance, num_products, card_val, active_val, salary
    ]])

    # 4. Scaling
    scaled_features = sc.transform(raw_features)

    # 5. Prediction
    prediction_prob = ann.predict(scaled_features)[0][0]
    
    # Using your requested logic: > 0.5 stays, else leaves
    is_staying = prediction_prob > 0.5

    # --- DISPLAY RESULTS ---
    st.divider()
    if is_staying:
        st.success(f"### Result: CUSTOMER STAYS")
        st.write(f"The model is {prediction_prob:.2%} confident the customer will remain.")
    else:
        st.error(f"### Result: CUSTOMER LEAVES")
        st.write(f"The model predicted a probability of {prediction_prob:.2%}, which is below the 50% threshold.")

    # Sidebar info for the students
    st.sidebar.info(f"Raw ANN Output: {prediction_prob:.4f}")