# Employee Salary Prediction API

A production-ready FastAPI web service for predicting employee salary class (<=50K or >50K) using a trained CatBoost model. Deployed on Render.com with CI/CD and custom domain support.

## 🚀 Features
- REST API for salary prediction
- Loads a pre-trained CatBoost model (`catboost_salary_model.joblib`)
- Input validation using Pydantic
- Returns both prediction and probability
- Interactive API docs at `/docs`

## 🛠️ Tech Stack
- FastAPI
- CatBoost
- Render.com (deployment)
- Python 3.8+

## 📦 Files
- `app.py` — FastAPI app
- `requirements.txt` — Python dependencies
- `Procfile` — Render.com process definition
- `catboost_salary_model.joblib` — Trained model
- `model_columns.joblib` — Model feature columns

## 🔗 API Usage
### Endpoint
`POST /predict`

### Request Body (JSON)
```json
{
  "age": 39,
  "workclass": "State-gov",
  "fnlwgt": 77516,
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "gender": "Male",
  "native-country": "United-States",
  "educational-num": 13,
  "capital-gain": 2174,
  "capital-loss": 0,
  "hours-per-week": 40
}
```
*Note: All model columns are required. Use the correct types and values as in training data.*

### Example Response
```json
{
  "prediction": 0,
  "probability": 0.13
}
```
- `prediction`: 0 = "<=50K", 1 = ">50K"
- `probability`: Probability of class ">50K"

## 🖥️ Local Development
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app locally:
   ```bash
   uvicorn app:app --reload
   ```
3. Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

## ☁️ Deployment (Render.com)
- Push all files to GitHub.
- Create a new Web Service on Render.com, connect your repo.
- Render will auto-detect `requirements.txt` and `Procfile`.
- Set the start command to:
  ```
  uvicorn app:app --host 0.0.0.0 --port 10000
  ```
- (Optional) Set up a custom domain and HTTPS.

## 📄 License
MIT 