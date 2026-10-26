# Store Sales Forecasting - Kaggle Time Series Project

This project predicts daily sales for product families sold at Favorita stores in Ecuador using the Kaggle **Store Sales - Time Series Forecasting** dataset.

The goal was to build a complete machine learning forecasting pipeline, including exploratory data analysis, feature engineering, model comparison, explainability, and validation using RMSLE.

## Project Overview

The dataset contains more than 3 million training rows across:

* 54 stores
* 33 product families
* Daily sales from 2013 to 2017
* Promotions
* Store metadata
* Oil prices
* Holidays and national events

The prediction target is daily sales for each store-product family combination.

## Methods

The project includes:

* Exploratory Data Analysis
* Time-based validation split
* Date and calendar features
* Store metadata features
* Oil price features
* Holiday and payday features
* Earthquake event feature
* Lag features
* Rolling mean, median, min, max, and standard deviation features
* EWMA features
* Yearly seasonality features
* XGBoost modeling
* CatBoost and LightGBM comparison
* Hyperparameter tuning
* SHAP explainability
* Weighted XGBoost ensemble

## Experimental Results

### Previous model experiments

The following results are from earlier model experiments. They are retained to document the progression of feature engineering and model development.

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

### Hyperparameter tuning and final test evaluation

The hyperparameter-tuning experiment used a chronological split, with July–December 2016 used for validation and January–August 15, 2017 reserved for final testing.

| Model | Evaluation set | RMSLE |
|---|---|---:|
| Seven-day lag baseline | Final test (2017) | 0.727332 |
| Tuned XGBoost | Final test (2017) | 0.462110 |

The tuned XGBoost model achieved a **36.46% reduction in RMSLE** compared with the seven-day lag baseline.

Hyperparameters selected using validation RMSLE:

- `n_estimators`: 700
- `max_depth`: 8
- `learning_rate`: 0.03
- `subsample`: 0.7
- `colsample_bytree`: 0.8

The final model was trained on a reproducible random sample of 500,000 rows from the pre-2017 development data. The 2017 test period was held out from hyperparameter selection.

The previous model results and the new tuned-model result come from different experiments and evaluation stages; they should not be treated as a direct, like-for-like comparison without verifying that their data splits and evaluation procedures match.

## Key Findings

* Sales are highly right-skewed.
* Around one-third of observations have zero sales.
* Promotions are strongly associated with higher sales.
* Weekend sales are higher than weekday sales.
* Product family and store identity are important.
* Recent sales history dominates model performance.
* Rolling mean features, particularly rolling_mean_7, were among the engineered predictors used to capture recent sales trends.

## Repository Structure

```text
notebooks/
├── 01_EDA.ipynb
├── 02_feature_engineering.ipynb
├── 03_baseline_model.ipynb
├── 04_xgboost_model.ipynb
├── 05_advanced_time_series_features.ipynb
├── 06_xgboost_with_lags.ipynb
├── 07_model_explainability_shap.ipynb
├── 08_hyperparameter_tuning.ipynb
├── 11_expanded_time_series_features.ipynb
├── 12_xgboost_expanded_features.ipynb
├── 13_catboost_expanded_features.ipynb
├── 15_xgboost_ewma_features.ipynb
├── 16_lightgbm_expanded_features.ipynb
├── 18_xgboost_seasonality_features.ipynb
└── 22_xgboost_ensemble.ipynb
```

## Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* SHAP

## Conclusion

This project demonstrates an end-to-end retail sales forecasting workflow, from exploratory data analysis and time-series feature engineering to model comparison, explainability, and hyperparameter tuning. Earlier experiments achieved a validation RMSLE of **0.4316** with a weighted XGBoost ensemble. In a separate hyperparameter-tuning experiment, the tuned XGBoost model achieved a final test RMSLE of **0.462110**, improving on the seven-day lag baseline by **36.46%**. These results highlight the value of historical sales features and the importance of evaluating forecasting models using chronological splits.
