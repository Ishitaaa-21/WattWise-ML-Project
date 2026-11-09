# WattWise — Smart Electricity Intelligence & Energy Saver

A beginner-friendly Machine Learning mini project built with Python, Streamlit, Pandas and Scikit-learn.

## Features
- Daily electricity usage visualization
- Linear Regression, Ridge Regression and Random Forest comparison
- MSE, RMSE, MAE and R² evaluation
- Next-day estimate demonstration
- Isolation Forest unusual-usage detection
- Estimated electricity cost calculator
- Energy-saving suggestions

## Requirements
- Python 3.10 or later recommended
- Internet connection for automatic UCI dataset download
- Alternatively, download the UCI dataset and upload a CSV/TXT file in the app

## Setup (Windows)
1. Install Python from https://www.python.org/downloads/
2. Extract this ZIP folder.
3. Open the extracted folder in VS Code.
4. Open Terminal in VS Code.
5. Create a virtual environment:

   `py -m venv .venv`

6. Activate it:

   `.venv\Scripts\activate`

7. Install packages:

   `python -m pip install -r requirements.txt`

8. Start the app:

   `python -m streamlit run app.py`

9. A browser tab should open. Click **Load open-access UCI dataset**.

## Dataset
UCI Individual Household Electric Power Consumption:
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The app downloads the public dataset automatically when internet access is available.

## Important academic note
This is a teaching/demo application, not a utility-grade forecast. The daily energy estimate is derived from average active power (kW) multiplied by 24 hours. The model comparison uses lagged daily consumption values (previous 1 and 7 days, plus rolling averages) to predict a later day. The train/test split is chronological to reduce future-data leakage. Results are still an academic baseline and may not generalize to every household.

## Suggested report sections
1. Abstract
2. Problem statement and objectives
3. Dataset description
4. Data preprocessing and feature engineering
5. Algorithms: Linear Regression, Ridge, Random Forest, Isolation Forest
6. Train/test split and metrics
7. Screenshots of dashboard and charts
8. Results and discussion
9. Limitations and future scope
10. Conclusion
