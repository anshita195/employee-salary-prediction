import streamlit as st
import requests

st.set_page_config(page_title="Employee Salary Prediction Dashboard", layout="centered")
st.title("Employee Salary Prediction Dashboard")

st.write("Enter employee details below to predict salary class (<=50K or >50K):")

# Define the input fields (update as per your model_columns)
def get_user_input():
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Local-gov", "Unknown", "State-gov", "Self-emp-inc", "Federal-gov"
    ])
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=50000)
    marital_status = st.selectbox("Marital Status", [
        "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    occupation = st.selectbox("Occupation", [
        "Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty", "Other-service", "Sales", "Craft-repair", "Transport-moving", "Unknown", "Machine-op-inspct", "Farming-fishing", "Tech-support", "Protective-serv", "Armed-Forces", "Priv-house-serv"
    ])
    relationship = st.selectbox("Relationship", [
        "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"
    ])
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ])
    gender = st.selectbox("Gender", ["Male", "Female"])
    native_country = st.selectbox("Native Country", [
        "United-States", "Mexico", "Philippines", "Germany", "Canada", "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", "Jamaica", "Unknown"
    ])
    educational_num = st.number_input("Educational Num", min_value=1, max_value=16, value=10)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
    return {
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

user_input = get_user_input()

if st.button("Predict Salary Class"):
    with st.spinner("Predicting..."):
        try:
            api_url = "https://employee-salary-prediction-5v1d.onrender.com/predict"
            response = requests.post(api_url, json=user_input, timeout=15)
            if response.status_code == 200:
                result = response.json()
                pred = result["prediction"]
                proba = result["probability"]
                label = ">50K" if pred == 1 else "<=50K"
                st.success(f"Prediction: {label}")
                st.info(f"Probability of >50K: {proba:.2%}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.caption("Powered by FastAPI + Streamlit | Deployed on Render.com") 