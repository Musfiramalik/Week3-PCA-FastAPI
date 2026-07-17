# Steel Industry Energy Consumption — Week 3 Task

## Overview
This project builds on Week 2's baseline regression model by applying Principal Component Analysis (PCA) for dimensionality reduction, and then deploying the model as an interactive web application using FastAPI. The task is divided into two parts, each with its own deliverable.

## Dataset & Model
- **Source:** The preprocessed and engineered dataset from Week 2 (Steel Industry Energy Consumption Dataset)
- **Base Model:** Random Forest Regressor (best performing model selected via cross-validation in Week 2)
---

## Part 1 — Dimensionality Reduction with PCA
**Notebook:** `notebooks/week3_pca.ipynb`

### What was done
- Loaded the Week 2 engineered dataset and applied identical preprocessing (dropping `date` and `High_Load`, one-hot encoding categorical features)
- Used the same 80/20 train-test split (`random_state=42`) as Week 2
- Fit `StandardScaler` and `PCA` on the training set only, then transformed both train and test sets (no data leakage)
- Applied PCA with all 28 components to inspect variance distribution
- Generated a scree plot and a cumulative explained variance curve
- Identified that **19 out of 28 components** are needed to reach 95% cumulative variance
- Retrained the Random Forest model using: (a) only 3 PCA components, (b) 19 components (95% variance), and (c) the original 28 features
- Compared MAE, RMSE, and R² across all three versions
- Created a loading heatmap showing how original features contribute to the top 3 principal components
- Saved the trained pipeline (scaler, PCA, models, and column order) using `joblib` for use in Part 2

### Results

| Model Version | MAE | RMSE | R² |
|---|---|---|---|
| Original (No PCA — 28 features) | 0.336 | 1.028 | 0.9991 |
| 3 PCA Components | 2.829 | 5.759 | 0.9704 |
| 19 PCA Components (95% Variance) | 1.722 | 3.561 | 0.9887 |

### Dimensionality Reduction Report
Reducing to 3 components caused a significant accuracy drop (RMSE increased ~460%), since too much signal is discarded. Using 19 components (95% variance) preserved most of the original accuracy (R² of 0.989 vs. 0.999) while removing about a third of the features. PCA could be considered for memory-constrained deployments where some accuracy trade-off is acceptable, but for this dataset the original full-feature Random Forest remains the stronger choice when accuracy is the priority.

---

## Part 2 — FastAPI Dashboard
**App:** `main.py`

### What was built
- **Home page (`/`)** — Welcome page with a navigation bar linking to Dashboard and Prediction pages
- **Dashboard page (`/dashboard`)** — Displays 3 key visualizations from Week 2's EDA:
  - Average Energy Usage by Hour of Day
  - Average Energy Consumption by Load Type
  - Correlation Heatmap
- **Prediction page (`/predict`)** — A form collecting all 12 raw input features (power readings, hour, day, load type, month, etc.), which are one-hot encoded, aligned to the model's expected columns, scaled, and passed to the trained Random Forest model to return a real-time energy consumption prediction

### How it works
1. The saved pipeline (`scaler.joblib`, `model.joblib`, `model_columns.joblib`) is loaded once when the app starts
2. User-submitted form values are converted into a single-row DataFrame
3. Categorical fields are one-hot encoded identically to training
4. Columns are reindexed to match the exact order and set used during training (missing dummy columns filled with 0)
5. The scaler transforms the input, and the Random Forest model predicts `Usage_kWh`

## How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open your browser at:http://127.0.0.1:8000
## Author
Musfira Malik — AI/ML Internship
