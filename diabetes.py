import streamlit as st
import numpy as np
import joblib


base = "dark"
primaryColor = "#1DB954"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"



hide_streamlit_cloud_elements = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    a[title="View source"] {display: none !important;}
    button[kind="icon"] {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_cloud_elements, unsafe_allow_html=True)
# Load the pre-trained model
model = joblib.load("diabetes.pkl")

def main():
    st.title("Diabetes Prediction App")

    pregnancies = st.slider("Number of Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
    insulin = st.slider("Insulin Level", 0, 846, 100)
    bmi = st.slider("BMI", 0.0, 67.12, 25.0)
    diabetes_pedigree_function = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.5)
    age = st.slider("Age", 21, 81, 30)

    # Button to trigger prediction
    if st.button("Predict"):
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        input_data = np.array(input_data).reshape(1, -1)  # Reshaping for a single prediction
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0] == 1:
            st.success("The person is diabetic.")
        else:
            st.success("The person is not diabetic.")

if __name__ == "__main__":
    main()
