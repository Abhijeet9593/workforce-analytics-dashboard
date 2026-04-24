import streamlit as st
import pandas as pd
import pickle

# Cache the model to keep the app fast
@st.cache_resource
def load_model():
    with open('collabera_hr_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

st.set_page_config(page_title="Talent Retention AI", page_icon="🏢")

st.title("🏢 Talent Retention & Attrition Predictor")
st.write("An HR Analytics model tailored for workforce management and staffing companies.")
st.markdown("---")

# Layout the input fields cleanly
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Employee Age", min_value=18, max_value=65, value=30)
    income = st.number_input("Monthly Income (USD)", min_value=1000, max_value=20000, value=5000)
    satisfaction = st.slider("Job Satisfaction Rating", min_value=1, max_value=4, value=3, help="1=Low, 4=Very High")

with col2:
    years = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
    distance = st.number_input("Distance From Home (miles)", min_value=1, max_value=30, value=5)
    overtime = st.selectbox("Works OverTime?", ["No", "Yes"])

st.markdown("---")

if st.button("Predict Employee Attrition", type="primary"):
    # Encode Overtime exactly how the model expects it
    ot_encoded = 1 if overtime == "Yes" else 0
    
    # Create the DataFrame for prediction
    input_data = pd.DataFrame([[age, income, satisfaction, years, ot_encoded, distance]], 
                              columns=['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany', 'OverTime', 'DistanceFromHome'])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Flight!** This employee shows signs of leaving. Intervention is recommended.")
    else:
        st.success("✅ **Low Risk.** This employee is likely to be retained.")
