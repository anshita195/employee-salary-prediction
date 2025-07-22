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
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=50000, help="Census weight (proxy for population)")
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
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ], help="Race")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Gender")
    native_country = st.selectbox("Native Country", [
        "United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", "Jamaica", "Unknown"
    ], help="Country of origin")
    educational_num = st.number_input("Educational Num", min_value=1, max_value=16, value=10, help="Number of years of education")
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, help="Capital gain (USD)")
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, help="Capital loss (USD)")
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40, help="Average hours worked per week")

user_input = {
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
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
        Made with ❤️ by <a href='https://github.com/anshita195' target='_blank'>Anshita</a> | Powered by CatBoost & Streamlit
    </div>
""", unsafe_allow_html=True) 