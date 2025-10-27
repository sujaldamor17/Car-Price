import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Call Streamlit page config early to avoid errors when using other st.* calls before config
st.set_page_config(page_title="Car Price Classifier 🚗", layout="centered")

# --- Load Model & Scaler ---
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "car_price_classifier.pkl"
scaler_path = BASE_DIR / "scaler.pkl"

model = None
scaler = None
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model from {model_path}: {e}")

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Failed to load scaler from {scaler_path}: {e}")

# Stop execution if loading failed
# --- Streamlit Page Config ---
st.title("🚘 Car Price Classification (Cheap vs Expensive)")
st.write("Predict whether a car is **Cheap (0)** or **Expensive (1)** based on its specifications.")
st.set_page_config(page_title="Car Price Classifier 🚗", layout="centered")
st.title("🚘 Car Price Classification (Cheap vs Expensive)")
st.write("Predict whether a car is **Cheap (0)** or **Expensive (1)** based on its specifications.")

# --- Sidebar Inputs ---
st.sidebar.header("Enter Car Specifications")

enginesize = st.sidebar.slider("Engine Size (cc)", 60, 350, 130)
horsepower = st.sidebar.slider("Horsepower", 40, 300, 100)
curbweight = st.sidebar.slider("Curb Weight (kg)", 1000, 4000, 2500)
citympg = st.sidebar.slider("City MPG", 5, 60, 30)
highwaympg = st.sidebar.slider("Highway MPG", 5, 60, 35)
wheelbase = st.sidebar.slider("Wheel Base", 85, 120, 100)
# added missing inputs that were referenced later
carlength = st.sidebar.slider("Car Length", 120, 250, 170)
carwidth = st.sidebar.slider("Car Width", 55, 90, 65)

# --- Prepare Input DataFrame ---
input_data = pd.DataFrame({
    'enginesize': [enginesize],
    'horsepower': [horsepower],
    'curbweight': [curbweight],
    'citympg': [citympg],
    'highwaympg': [highwaympg],
    'wheelbase': [wheelbase],
    'carlength': [carlength],
    'carwidth': [carwidth]
})

# --- Scale Input ---
try:
    scaled_input = scaler.transform(input_data)
except Exception as e:
    st.error(f"Failed to scale input data: {e}")
    st.stop()

# --- Predict ---
try:
    prediction = model.predict(scaled_input)[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# compute probability if available and available for the predicted class
probability = None
if hasattr(model, "predict_proba"):
    try:
        proba = model.predict_proba(scaled_input)[0]
        if hasattr(model, "classes_"):
            # find index of predicted class in classes_
            try:
                idx = list(model.classes_).index(prediction)
                probability = proba[idx]
            except ValueError:
                # fallback: if classes_ don't contain prediction as expected
                probability = None
        else:
            # fallback for integer labels aligning with proba indices
            if isinstance(prediction, (int, np.integer)) and prediction < len(proba):
                probability = proba[prediction]
    except Exception:
        probability = None

# --- Show Result ---
st.subheader("🔍 Prediction Result:")
if probability is not None:
    prob_str = f" (Probability: {probability:.2f})"
else:
    prob_str = ""

if prediction == 1:
    st.success(f"💰 The car is **Expensive**{prob_str}")
else:
    st.info(f"🚗 The car is **Cheap**{prob_str}")

st.markdown("---")
st.caption("Developed by Sujal Damor • Logistic Regression Model")
