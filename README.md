# Employee Salary Prediction System

This project implements a machine learning-based system for predicting employee salary class (<=50K or >50K) using demographic and work-related features. The system consists of two components: a FastAPI-based REST API service and an interactive Streamlit web dashboard. The model is trained using CatBoost classifier on the Adult Income dataset and achieves an accuracy of 87.17% with an ROC AUC score of 0.9229.

## Project Overview

The system provides two interfaces for salary prediction:
1. **REST API Service**: A production-ready FastAPI web service that accepts JSON requests and returns predictions
2. **Interactive Dashboard**: A user-friendly Streamlit web application that allows users to input employee details through a graphical interface

The model was developed using a comprehensive data science pipeline including data preprocessing, outlier handling, feature engineering, and model evaluation. The CatBoost algorithm was selected for its superior performance in handling categorical features and achieving high accuracy on the classification task.

## Features

### API Service
- RESTful API endpoint for salary prediction
- Input validation using Pydantic models
- Returns both binary prediction and probability scores
- Interactive API documentation available at `/docs` endpoint
- Production-ready deployment configuration

### Dashboard Application
- User-friendly web interface built with Streamlit
- Simplified input form requiring only essential fields (age, marital status, occupation, education level, gender)
- Automatic handling of additional model features with sensible defaults
- Real-time prediction with probability visualization
- Responsive design with intuitive user experience

## Technology Stack

- **FastAPI**: Web framework for building the REST API
- **Streamlit**: Framework for creating the interactive dashboard
- **CatBoost**: Gradient boosting library for model training and prediction
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical computations
- **Scikit-learn**: Model evaluation metrics and data splitting
- **Joblib**: Model serialization and loading
- **Pydantic**: Data validation for API requests
- **Python 3.8+**: Programming language

## Project Structure

```
.
├── app.py                          # FastAPI application
├── dashboard.py                    # Streamlit dashboard application
├── employee_salary.ipynb           # Jupyter notebook with complete data science pipeline
├── catboost_salary_model.joblib    # Trained CatBoost model
├── model_columns.joblib            # Model feature columns in correct order
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment configuration for Render.com
└── README.md                       # Project documentation
```

## Model Details

The model was trained on the Adult Income dataset from the UCI Machine Learning Repository. The training process involved:

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and removing redundant features
- **Outlier Handling**: Clipping numerical features at 1st and 99th percentiles to reduce the impact of extreme values
- **Feature Engineering**: Dropping the redundant 'education' column in favor of 'educational-num' for better ordinal representation
- **Model Training**: CatBoost classifier with 500 iterations, learning rate of 0.1, and depth of 6
- **Evaluation Metrics**: 
  - Accuracy: 87.17%
  - ROC AUC: 0.9229
  - F1 Score: 0.7183 (at optimal threshold of 0.4)

### Model Features

The model requires the following 12 features:
- `age`: Employee age (numerical)
- `workclass`: Type of employer (categorical)
- `marital-status`: Marital status (categorical)
- `occupation`: Occupation type (categorical)
- `relationship`: Relationship to household (categorical)
- `race`: Race (categorical)
- `gender`: Gender (categorical)
- `native-country`: Country of origin (categorical)
- `educational-num`: Education level encoded as number (numerical, 1-16)
- `capital-gain`: Capital gains in USD (numerical)
- `capital-loss`: Capital losses in USD (numerical)
- `hours-per-week`: Average hours worked per week (numerical)

## API Usage

### Endpoint
```
POST /predict
```

### Request Body

The API requires all 12 model features in JSON format. The request body should include:

```json
{
  "age": 39,
  "workclass": "State-gov",
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

**Note**: All fields are required and must match the exact feature names and data types used during model training. Categorical values must match the training data categories exactly.

### Response Format

```json
{
  "prediction": 0,
  "probability": 0.13
}
```

- `prediction`: Binary classification result (0 = "<=50K", 1 = ">50K")
- `probability`: Probability score for the ">50K" class (ranges from 0.0 to 1.0)

### Error Handling

The API returns appropriate HTTP status codes:
- `200`: Successful prediction
- `400`: Bad request (missing or invalid input fields)
- `500`: Internal server error

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the API Service

1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```

2. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. Test the API endpoint:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 39,
       "workclass": "State-gov",
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
     }'
```

### Running the Dashboard Application

1. Start the Streamlit application:
```bash
streamlit run dashboard.py
```

2. Access the dashboard in your web browser:
   - Default URL: http://localhost:8501

The dashboard provides a simplified interface where users only need to enter:
- Age
- Marital Status
- Occupation
- Education Level (dropdown with names mapped to numbers)
- Gender

All other required model features are automatically set to sensible default values in the background.


## Data Science Pipeline

The complete data science workflow is documented in `employee_salary.ipynb`, which includes:

1. **Data Loading**: Loading the Adult Income dataset
2. **Data Exploration**: Initial analysis of data shape, distributions, and missing values
3. **Data Cleaning**: Handling missing values, replacing '?' with 'Unknown', removing irrelevant categories
4. **Outlier Detection and Removal**: Clipping numerical features at 1st and 99th percentiles
5. **Exploratory Data Analysis**: Visualizations of target distribution, feature distributions, and correlations
6. **Feature Engineering**: Dropping redundant columns and selecting relevant features
7. **Data Splitting**: Train-test split with stratification
8. **Model Training**: CatBoost classifier training with evaluation on test set
9. **Model Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and precision-recall curve
10. **Threshold Tuning**: Finding optimal classification threshold based on F1-score
11. **Feature Importance Analysis**: Identifying most important features for prediction
12. **Model Persistence**: Saving the trained model and feature columns for deployment

## Results

The final CatBoost model achieved the following performance metrics on the test set:

- **Accuracy**: 87.17%
- **ROC AUC**: 0.9229
- **Precision** (class >50K): 0.77
- **Recall** (class >50K): 0.65
- **F1-Score** (class >50K): 0.70
- **Best Threshold**: 0.4 (optimized for F1-score)

The model demonstrates strong performance in predicting salary classes, with particularly high accuracy for the majority class (<=50K) and reasonable performance for the minority class (>50K).

## Future Enhancements

Potential improvements and extensions for the system include:

- Integration of model explainability tools (SHAP values) to provide insights into prediction factors
- Expansion of the dataset with more recent and diverse samples to improve model generalization
- Deployment to cloud platforms with auto-scaling capabilities
- Implementation of model versioning and A/B testing infrastructure
- Addition of authentication and rate limiting for production use

## References

- UCI Machine Learning Repository: Adult Data Set. Available at: https://archive.ics.uci.edu/ml/datasets/adult
- CatBoost Documentation: https://catboost.ai/docs/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Streamlit Documentation: https://docs.streamlit.io/
- Scikit-learn Documentation: https://scikit-learn.org/

## License

This project is licensed under the MIT License.
