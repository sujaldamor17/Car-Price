import streamlit as st
import pandas as pd
import joblib

# ============================
# Load model, scaler, features
# ============================
model = joblib.load(r"C:\Users\Lenovo\Desktop\Log_regression\car_price_classifier.pkl")
scaler = joblib.load(r"C:\Users\Lenovo\Desktop\Log_regression\scaler.pkl")
trained_columns = joblib.load(r"C:\Users\Lenovo\Desktop\Log_regression\feature_names.pkl")

# ============================
# Streamlit UI
# ============================
st.title("🚗 Car Price Prediction App")
st.write("Enter car details below and click **Predict** to estimate price category.")

# === Input fields ===
symboling = st.number_input("Symboling", value=3)
fueltype = st.selectbox("Fuel Type", ["gas", "diesel"])
aspiration = st.selectbox("Aspiration", ["std", "turbo"])
doornumber = st.selectbox("Door Number", ["two", "four"])
carbody = st.selectbox("Car Body", ["convertible", "hatchback", "sedan", "wagon", "hardtop"])
drivewheel = st.selectbox("Drive Wheel", ["rwd", "fwd", "4wd"])
enginelocation = st.selectbox("Engine Location", ["front", "rear"])
wheelbase = st.number_input("Wheelbase", value=90.0)
curbweight = st.number_input("Curb Weight", value=2500)
enginesize = st.number_input("Engine Size", value=130)
horsepower = st.number_input("Horsepower", value=100)
citympg = st.number_input("City MPG", value=25)
highwaympg = st.number_input("Highway MPG", value=30)

# === Create DataFrame from inputs ===
input_df = pd.DataFrame({
    "symboling": [symboling],
    "fueltype": [fueltype],
    "aspiration": [aspiration],
    "doornumber": [doornumber],
    "carbody": [carbody],
    "drivewheel": [drivewheel],
    "enginelocation": [enginelocation],
    "wheelbase": [wheelbase],
    "curbweight": [curbweight],
    "enginesize": [enginesize],
    "horsepower": [horsepower],
    "citympg": [citympg],
    "highwaympg": [highwaympg]
})

# === One-hot encode categorical columns (same as training) ===
input_encoded = pd.get_dummies(input_df)

# === Reindex columns to match training features ===
input_encoded = input_encoded.reindex(columns=trained_columns, fill_value=0)

# === Scale the data ===
input_scaled = scaler.transform(input_encoded)

# === Prediction button ===
if st.button("Predict Price Category"):
    try:
        pred = model.predict(input_scaled)
        st.success(f"Predicted Car Price Category: **{pred[0]}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
