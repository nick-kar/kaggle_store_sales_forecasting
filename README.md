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

| Model                                   | RMSLE  |
|---------------------------------------- |-------:|
| Linear Regression                       | 2.1040 |
| XGBoost                                 | 0.6839 |
| XGBoost + Lags                          | 0.4573 |
| XGBoost + Expanded Features             | 0.4325 |
| CatBoost                                | 0.4518 |
| LightGBM                                | 0.4501 |
| XGBoost + EWMA + Seasonality            | 0.4320 |
| XGBoost Ensemble                        | 0.4316 |

The largest improvement came from adding store-family-level lag and rolling-window features, showing that recent sales history was the strongest predictive signal.
This comparison demonstrates that advanced feature engineering contributed more to performance gains than changing the underlying boosting algorithm. XGBoost consistently outperformed both CatBoost and LightGBM on this dataset.

## Key Findings

* Sales are highly right-skewed.
* Around one-third of observations have zero sales.
* Promotions are strongly associated with higher sales.
* Weekend sales are higher than weekday sales.
* Product family and store identity are important.
* Recent sales history dominates model performance.
* Rolling mean features, especially `rolling_mean_7` and `rolling_mean_3`, were the most important predictors.
* SHAP analysis confirmed that short-term store-family sales history drives most predictions.

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

This project demonstrates an end-to-end time-series machine learning workflow. The final weighted XGBoost ensemble achieved a validation RMSLE of **0.4316**, with most performance gains coming from advanced time-series feature engineering rather than model tuning alone.
