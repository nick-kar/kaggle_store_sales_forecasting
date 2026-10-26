# Store Sales Forecasting — Kaggle Time Series Project

This project predicts daily sales for product families sold at Favorita stores in Ecuador using the [Kaggle Store Sales - Time Series Forecasting dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).

The goal is to build an end-to-end machine learning forecasting workflow covering exploratory data analysis, time-series feature engineering, model comparison, explainability, hyperparameter tuning, and evaluation using Root Mean Squared Logarithmic Error (RMSLE).

## Project Overview

The dataset contains more than 3 million training observations covering:

- 54 stores
- 33 product families
- Daily sales from 2013 to 2017
- Promotions
- Store metadata
- Oil prices
- Holidays and national events

The prediction target is daily sales for each store-product family combination.

## Methods and Features

The project explores a range of approaches to retail sales forecasting:

- Exploratory Data Analysis (EDA)
- Chronological training, validation, and test splits
- Date and calendar features
- Store metadata features
- Oil price features
- Holiday and payday features
- Earthquake event features
- Lag features
- Rolling mean, median, minimum, maximum, and standard deviation features
- Exponentially Weighted Moving Average (EWMA) features
- Yearly seasonality features
- Target encoding and aggregate features
- XGBoost modeling
- CatBoost and LightGBM comparison
- Hyperparameter tuning
- SHAP model explainability
- Weighted XGBoost ensemble
- Recursive forecasting experiments

## Experimental Results

### Previous Model Experiments

The following results are from earlier model experiments. They document the progression of feature engineering and model development.

| Model | RMSLE |
|---|---:|
| Linear Regression | 2.1040 |
| XGBoost | 0.6839 |
| XGBoost + Lags | 0.4573 |
| XGBoost + Expanded Features | 0.4325 |
| CatBoost | 0.4518 |
| LightGBM | 0.4501 |
| XGBoost + EWMA + Seasonality | 0.4320 |
| XGBoost Ensemble | 0.4316 |

These results come from earlier experiments and should be interpreted in the context of their respective data splits and evaluation procedures.

### Hyperparameter Tuning and Final Test Evaluation

A separate hyperparameter-tuning experiment used chronological data splits:

- **Training/development period:** January 2013 – June 2016
- **Validation period:** July – December 2016
- **Final test period:** January 1 – August 15, 2017

Ten XGBoost parameter configurations were evaluated using a reproducible 500,000-row sample of the training data for the tuning process.

The resulting model was evaluated against a seven-day lag baseline on the held-out 2017 test period.

| Model | Evaluation Set | RMSLE |
|---|---|---:|
| Seven-day lag baseline | Final test (2017) | 0.727332 |
| Tuned XGBoost | Final test (2017) | 0.462110 |

The tuned XGBoost model achieved a **36.46% reduction in RMSLE** compared with the seven-day lag baseline.

The selected hyperparameters were:

- `n_estimators`: 700
- `max_depth`: 8
- `learning_rate`: 0.03
- `subsample`: 0.7
- `colsample_bytree`: 0.8

The final model was trained on a reproducible random sample of 500,000 rows from the pre-2017 development data. The 2017 test period was held out from hyperparameter selection.

The earlier model results and the tuned-model result belong to different experiments and evaluation stages. They should not be treated as a direct, like-for-like comparison unless their data splits and evaluation procedures are verified to be equivalent.

## Key Findings

- Sales are highly right-skewed.
- Approximately one-third of observations have zero sales.
- Promotions are strongly associated with higher sales.
- Weekend sales are higher than weekday sales in the analyzed data.
- Product family and store identity are important predictive features.
- Recent sales history is an important source of predictive information.
- Rolling features, particularly `rolling_mean_7`, were used to capture recent sales trends.
- Chronological evaluation is important for assessing forecasting models without using future observations during training.

## Repository Structure

```text
kaggle_store_sales_forecasting/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_xgboost_model.ipynb
│   ├── 05_advanced_time_series_features.ipynb
│   ├── 06_xgboost_with_lags.ipynb
│   ├── 07_model_explainability_shap.ipynb
│   ├── 08_hyperparameter_tuning.ipynb
│   ├── 09_final_submission.ipynb
│   ├── 10_recursive_forecasting_submission.ipynb
│   ├── 11_expanded_time_series_features.ipynb
│   ├── 12_xgboost_expanded_features.ipynb
│   ├── 13_catboost_expanded_features.ipynb
│   ├── 14_ewma_features.ipynb
│   ├── 15_xgboost_ewma_features.ipynb
│   ├── 16_lightgbm_expanded_features.ipynb
│   ├── 17_yearly_seasonality_features.ipynb
│   ├── 18_xgboost_seasonality_features.ipynb
│   ├── 19_target_encoding_and_aggregates.ipynb
│   ├── 20_xgboost_target_encoding.ipynb
│   ├── 21_final_recursive_best_model.ipynb
│   └── 22_xgboost_ensemble.ipynb
├── README.md
└── requirements.txt
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nick-kar/kaggle_store_sales_forecasting.git
cd kaggle_store_sales_forecasting
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

The project uses Python libraries including Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost, LightGBM, CatBoost, and SHAP.

### 4. Download the Dataset

Download the **Store Sales - Time Series Forecasting** dataset from Kaggle:

https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

Follow Kaggle's dataset access requirements and terms of use. Place the required CSV files in the locations expected by the notebooks. Check the file paths in each notebook and adjust them to match your local directory structure if necessary.

The dataset is not included in this repository.

### 5. Run the Notebooks

Open the project in Jupyter Notebook or Visual Studio Code.

For an initial exploration, start with `01_EDA.ipynb`. Then review the feature-engineering and modeling notebooks, including:

- `02_feature_engineering.ipynb` — feature preparation
- `03_baseline_model.ipynb` — baseline modeling
- `06_xgboost_with_lags.ipynb` — lag-based forecasting features
- `07_model_explainability_shap.ipynb` — model explainability
- `08_hyperparameter_tuning.ipynb` — hyperparameter tuning and final test evaluation
- `13_catboost_expanded_features.ipynb` — CatBoost experiment
- `16_lightgbm_expanded_features.ipynb` — LightGBM experiment
- `21_final_recursive_best_model.ipynb` — recursive forecasting experiment
- `22_xgboost_ensemble.ipynb` — weighted XGBoost ensemble

Not every notebook is necessarily independent. Some experiments may rely on intermediate data or features created by earlier notebooks. Check each notebook's inputs and file paths before running it independently.

## Tools Used

- **Programming:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine learning:** Scikit-learn, XGBoost, CatBoost, LightGBM
- **Explainability:** SHAP
- **Development environment:** Jupyter Notebook, Visual Studio Code
- **Version control:** Git and GitHub
- **Dataset:** Kaggle Store Sales - Time Series Forecasting

## Conclusion

This project demonstrates an end-to-end retail sales forecasting workflow, from exploratory data analysis and time-series feature engineering to model comparison, explainability, and hyperparameter tuning.

Earlier experiments achieved an RMSLE of **0.4316** with a weighted XGBoost ensemble. In a separate hyperparameter-tuning experiment, the tuned XGBoost model achieved a final test RMSLE of **0.462110**, improving on the seven-day lag baseline by **36.46%**.

Together, these experiments demonstrate the value of historical sales features, systematic model comparison, and chronological evaluation when developing machine learning solutions for retail forecasting.