import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

# Load the trained model using joblib
model = joblib.load('voting_with_advanced.pkl')

# Streamlit app title and GitHub link
st.set_page_config(page_title="Lightning Strike Prediction", layout="centered")

# Attempt to load the GitHub icon
icon_path = "github_icon.png"
if os.path.exists(icon_path):
    github_icon = Image.open(icon_path)
    st.image(github_icon, width=30)
else:
    st.write("🔗 [Developer's GitHub](https://github.com/hamza1713)")

# Streamlit app title
st.title("Lightning Strike Prediction")

# Input features
st.write("### Input Features")

x_coord = st.number_input("Enter x_coord:", value=0.0)
y_coord = st.number_input("Enter y_coord:", value=0.0)
distance_from_center = st.number_input("Enter distance_from_center:", value=0.0)

# Prediction button
if st.button("Predict"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'x_coord': [x_coord],
        'y_coord': [y_coord],
        'distance_from_center': [distance_from_center]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display prediction with one decimal place
    st.write(f"### Prediction: {prediction[0]:.1f}")
