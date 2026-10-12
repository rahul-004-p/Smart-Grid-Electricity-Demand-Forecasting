# Smart Grid Electricity Demand Forecasting

**Machine Learning • Electricity Demand Analysis • Power BI Dashboard**

## Project Overview

This project analyzes historical electricity demand in England and Wales and develops a machine-learning workflow for day-ahead electricity-demand forecasting. It combines exploratory data analysis (EDA), feature engineering, model comparison, evaluation, and an interactive Power BI dashboard.

## Objectives

- Analyze historical electricity-demand patterns across hours, days, and seasons.
- Examine wind and solar generation/capacity alongside electricity demand.
- Clean and prepare time-series data for machine learning.
- Engineer time-based and historical-demand features.
- Compare regression models and select a forecasting model.
- Present demand trends and key indicators in Power BI.

## Dataset

The project presentation describes the source data as England and Wales electricity data covering **2021–2025**, with **86,976 half-hourly records** and **27 raw columns**. The target variable is `Electricity_Demand` (MW).

**Note:** The raw CSV dataset is not included among the files currently supplied with this project. To run the notebooks, obtain the permitted source dataset and place it at the file path expected by the first notebook.

## Workflow

1. **EDA and feature engineering** — check missing values and duplicates, construct a timestamp from date and half-hour period, explore demand patterns, and create model features.
2. **Model selection** — use a chronological train/test split, with the final 90 days reserved for testing, and compare Linear Regression, Random Forest, and Gradient Boosting.
3. **Final model** — retrain the selected model, evaluate it on the held-out period, visualize actual versus predicted demand, and inspect feature importance when supported.
4. **Power BI dashboard** — explore demand KPIs and trends by hour, day, month, and year, including renewable-generation comparisons.

## Features Used

The model uses 13 features:

- Cyclical time features: `Hour_sin`, `Hour_cos`, `Month_sin`, `Month_cos`, `DoW_sin`, `DoW_cos`
- Calendar features: `Is_Weekend`, `Quarter`, `Year`
- Renewable capacity features: `Wind_Capacity`, `Solar_Capacity`
- Historical demand features: `Demand_Previous_24hr`, `Demand_Previous_7days`

Target: `Electricity_Demand`.

## Models Compared

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor (implemented with scikit-learn's `HistGradientBoostingRegressor`)

The presentation reports the following test-set results:

| Model | MAE (MW) | RMSE (MW) | MAPE |
|---|---:|---:|---:|
| Random Forest (selected) | 1,334.7 | 1,817.0 | 5.00% |
| Linear Regression | 1,353.4 | 1,763.8 | 5.15% |
| Gradient Boosting | 1,564.0 | 2,153.0 | 5.65% |

The presentation reports **95.00% accuracy for Random Forest**, calculated as `100 − MAPE`. This is the project's own MAPE-derived indicator, not classification accuracy.

The selected Random Forest configuration is `n_estimators=300` and `max_depth=14`.

## Key Findings

As summarized in the project presentation:

- Average demand peaks around 18:00 (6 PM) and is lowest overnight near 4 AM.
- Demand is higher on weekdays than weekends.
- Demand is highest in winter months and lowest in summer months.
- Solar generation peaks around midday, before the evening demand peak.
- Demand from 24 hours earlier is an important predictor, followed by demand from 7 days earlier.

## Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- Microsoft Power BI

## Project Files

- `01_EDA_Feature_Engineering.ipynb` — data cleaning, EDA, and feature engineering.
- `02_Model_Selection.ipynb` — time-based split and model comparison.
- `03_Final_Model (1).ipynb` — final training, evaluation, and predictions.
- `el.pbix` — Power BI dashboard file.
- `SmartGrid_Electricity_Demand_Forecasting.pptx` — project presentation.

## How to Run the Notebooks

1. Install Python and Jupyter Notebook, or use a compatible notebook environment.
2. Install the libraries used by the notebooks: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `joblib`.
3. Place the source dataset at the path referenced in Notebook 1: `England_Wales_Electricity_2021_2025_final.csv`.
4. Run Notebook 1 to create the cleaned/feature-engineered dataset.
5. Run Notebook 2 to compare models and save the selected-model metadata.
6. Run Notebook 3 after the required dataset and model-selection outputs have been created.

The notebooks use relative file paths for intermediate files and outputs. If you move the notebooks into different folders, update those paths accordingly. The dataset and any generated intermediate files must be available for the workflow to run end-to-end.

## Dashboard

Open `el.pbix` in Microsoft Power BI Desktop to view the dashboard. The presentation describes demand KPI cards, hour/day breakdowns, renewable generation versus demand, and month/year filters.

## Author

**Rahul P**  
Data Analytics / Machine Learning Project
