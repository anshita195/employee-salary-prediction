import streamlit as st
import joblib
import pandas as pd

# Load model and columns once
@st.cache_resource
def load_model():
    model = joblib.load('catboost_salary_model.joblib')
    columns = joblib.load('model_columns.joblib')
    return model, columns

model, model_columns = load_model()

st.set_page_config(page_title="Employee Salary Prediction Dashboard", layout="centered")

st.markdown("""
    <div style='text-align:center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='100'/>
    </div>
    <h1 style='text-align:center; color:#4F8BF9;'>Employee Salary Prediction Dashboard</h1>
    <p style='text-align:center;'>Enter employee details below to predict salary class (<=50K or >50K):</p>
    <hr>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Employee's age (18-100)")
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Local-gov", "Unknown", "State-gov", "Self-emp-inc", "Federal-gov"
    ], help="Type of employer")
    marital_status = st.selectbox("Marital Status", [
        "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ], help="Marital status")
    occupation = st.selectbox("Occupation", [
        "Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty", "Other-service", "Sales", "Craft-repair", "Transport-moving", "Unknown", "Machine-op-inspct", "Farming-fishing", "Tech-support", "Protective-serv", "Armed-Forces", "Priv-house-serv"
    ], help="Occupation type")
    relationship = st.selectbox("Relationship", [
        "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"
    ], help="Relationship to household")

with col2:
    # Education mapping
    education_mapping = {
        'Preschool': 1,
        '1st-4th': 2,
        '5th-6th': 3,
        '7th-8th': 4,
        '9th': 5,
        '10th': 6,
        '11th': 7,
        '12th': 8,
        'HS-grad': 9,
        'Some-college': 10,
        'Assoc-voc': 11,
        'Assoc-acdm': 12,
        'Bachelors': 13,
        'Masters': 14,
        'Prof-school': 15,
        'Doctorate': 16
    }
    education_level = st.selectbox(
        "Education Level",
        options=list(education_mapping.keys()),
        help="Highest level of education completed"
    )
    educational_num = education_mapping[education_level]
    gender = st.selectbox("Gender", ["Male", "Female"], help="Gender")

# Set default values for hidden/advanced fields
race = "Other"  # default
native_country = "United-States"  # default
capital_gain = 0  # default
capital_loss = 0  # default
hours_per_week = 40  # default, typical full-time

user_input = {
    "age": age,
    "workclass": workclass,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "gender": gender,
    "native-country": native_country,
    "educational-num": educational_num,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week
}

st.markdown("<hr>", unsafe_allow_html=True)

if st.button("🎯 Predict Salary Class"):
    with st.spinner("Predicting..."):
        try:
            # Prepare input DataFrame in correct order
            X = pd.DataFrame([[user_input[col] for col in model_columns]], columns=model_columns)
            pred = model.predict(X)[0]
            proba = float(model.predict_proba(X)[0][1])
            label = ">50K" if pred == 1 else "<=50K"
            color = "#27ae60" if pred == 1 else "#e74c3c"
            st.markdown(f"<h2 style='color:{color};text-align:center;'>Prediction: {label}</h2>", unsafe_allow_html=True)
            st.progress(proba)
            st.info(f"Probability of >50K: {proba:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("""
    <hr>
    <div style='text-align:center;font-size:16px;'>
        Made by <a href='https://github.com/anshita195' target='_blank'>Anshita</a> | Streamlit
    </div>
""", unsafe_allow_html=True) 