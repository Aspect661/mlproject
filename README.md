# Student Math Score Predictor — End-to-End MLOps Project

A Flask web application that predicts a student's math score based on demographic and academic features, built with a full MLOps pipeline covering data ingestion, transformation, model training, and prediction.

## Problem Statement

Given a student's gender, race/ethnicity, parental level of education, lunch type, test preparation course status, reading score, and writing score — predict their math score.

## Project Structure

```
mlproject/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Loads data, splits into train/test
│   │   ├── data_transformation.py  # Preprocessing pipeline (scaling, encoding)
│   │   └── model_trainer.py        # Trains and evaluates multiple models
│   └── pipeline/
│       ├── train_pipeline.py        # Orchestrates training
│       └── predict_pipeline.py      # Loads artifacts, runs inference
├── exception/                       # Custom exception handler
├── logger/                          # Logging setup
├── utils/                           # Shared utilities (save/load objects, evaluate models)
├── notebook/                        # EDA and experimentation notebooks
│   └── data/stud.csv                # Raw dataset
├── artifacts/                       # Generated outputs (model, preprocessor, splits)
├── app.py                           # Flask web application
├── setup.py
└── requirements.txt
```

## ML Pipeline

### 1. Data Ingestion
Reads `stud.csv`, saves a raw copy, and splits into 80/20 train/test sets under `artifacts/`.

### 2. Data Transformation
Builds a `ColumnTransformer` preprocessing pipeline:
- **Numerical features** (`reading_score`, `writing_score`): median imputation + standard scaling
- **Categorical features** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`): most-frequent imputation + one-hot encoding + standard scaling

Serialises the fitted preprocessor to `artifacts/preprocessor.pkl`.

### 3. Model Training
Trains and evaluates 8 regression models via `GridSearchCV`:

| Model | Hyperparameters tuned |
|---|---|
| Random Forest | `n_estimators` |
| Decision Tree | `criterion` |
| Gradient Boosting | `learning_rate`, `subsample`, `n_estimators` |
| Linear Regression | — |
| K-Neighbours Regressor | `n_neighbors` |
| XGBoost Regressor | `learning_rate`, `n_estimators` |
| CatBoost Regressor | `depth`, `learning_rate`, `iterations` |
| AdaBoost Regressor | `learning_rate`, `n_estimators` |

The best model (by R² on the test set) is saved to `artifacts/trained_model.pkl`. Training aborts if the best R² is below 0.6.

## Setup & Usage

### 1. Clone and install
```bash
git clone <repo-url>
cd mlproject
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/components/data_ingestion.py
```
This runs the full pipeline (ingestion → transformation → training) and prints the final R² score.

### 3. Run the web app
```bash
python app.py
```
Navigate to `http://localhost:8000/predictdata` to use the prediction form.

## Tech Stack

- **ML**: scikit-learn, XGBoost, CatBoost
- **Web**: Flask
- **Serialisation**: dill
- **Data**: pandas, numpy
- **Logging & Exceptions**: custom modules (`logger/`, `exception/`)
