# Agent Handoff Specification: NYC Yellow Taxi Demand Forecasting

**Target Agent / System:** Antigravity  
**Project Title:** NYC Yellow Taxi Hourly Zone Demand Forecasting (Big Data Analytics)  
**Task Type:** Time-Series / Tabular Supervised Learning — Regression  
**Primary Objective:** Deliver a fully executed, end-to-end Jupyter Notebook (`.ipynb`) satisfying all rubric criteria for the course "Part 2: Data Management and Machine Learning Pipeline".

---

## 1. Project Context & Requirements Baseline

### 1.1 Business Problem & Purpose
- **Context:** Yellow Taxis in NYC experience heavy fluctuations in passenger demand across zones and times of day.
- **Core Question:** Can we accurately predict the hourly taxi pickup demand (`pickup_count`) in each **Yellow Zone** based on spatial, temporal, and seasonal attributes?
- **Stakeholders & Value:**
  1. **Fleet Dispatch Operators:** Optimize fleet rebalancing to high-demand zones.
  2. **Taxi Drivers:** Reduce deadheading time and locate high-yield pickup zones by hour.
  3. **Urban Planners (NYC TLC):** Manage congestion pricing strategies and taxi stand allocations.

### 1.2 Data Unit of Analysis & Target Definition
- **Unit of Analysis (1 Row):** **1 Zone x 1 Hourly Time Window** (`PULocationID` + `pickup_hour`).
- **Target Variable:** **`pickup_count`** (Integer count of trips originating in that zone within that hour).
- **Filtering Constraint:** Strictly filter for `PU_service_zone == 'Yellow Zone'`.

---

## 2. Dataset Specifications & Ingestion Logic

### 2.1 Raw Data Sources
1. **Trip Records:** 2024 Yellow Taxi Parquet files from NYC TLC.
   - Sampling strategy: Full 7-day blocks (8th–14th of Jan, Apr, Jul, Oct 2024) representing 4 seasons (~3.1M raw trip rows total).
2. **Zone Lookup Table:** `taxi_zone_lookup.csv` (contains `LocationID`, `Borough`, `Zone`, `service_zone`).

### 2.2 Data Ingestion & Pre-Processing Steps
```python
import os
import urllib.request
import pandas as pd

# Load seasonal 7-day blocks
seasonal_blocks = [
    ('2024-01', 'Winter', '2024-01-08', '2024-01-15'),
    ('2024-04', 'Spring', '2024-04-08', '2024-04-15'),
    ('2024-07', 'Summer', '2024-07-08', '2024-07-15'),
    ('2024-10', 'Fall',   '2024-10-08', '2024-10-15'),
]

# Merge zone metadata & filter
# PU_service_zone == 'Yellow Zone'
# Floor pickup datetime to hourly frequency: df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')

# Aggregate to Hourly Zone Demand Target:
# demand_df = df.groupby(['PULocationID', 'PU_Borough', 'PU_Zone', 'season', 'pickup_hour']).size().reset_index(name='pickup_count')
```

---

## 3. Machine Learning Pipeline Execution Plan

Antigravity must generate and execute code cells for Sections 7 through 13 in the target notebook template according to the following guidelines:

### Section 7: Feature Engineering
Build leakage-safe features available prior to prediction time:
- **Temporal Features:** `hour` (0–23), `day_of_week` (0–6), `is_weekend` (0/1), `month` (1, 4, 7, 10), `season`.
- **Cyclical Time Encoding:** `hour_sin`, `hour_cos` using $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$.
- **Spatial / Categorical:** `PULocationID`, `PU_Borough`.
- *(Note: Avoid complex lag features across non-contiguous seasonal block boundaries to prevent missing value artifacts).*

### Section 8: Data Partitioning (Time-based Split)
- **Constraint:** Strictly avoid random shuffle (`train_test_split(shuffle=True)`) to eliminate time-series data leakage.
- **Train Set:** Seasons 1–3 (Winter, Spring, Summer: Jan, Apr, Jul 2024).
- **Test Set:** Season 4 (Fall: Oct 2024).

### Section 9: Preprocessing Pipeline
- **Categorical Processing:** OneHotEncoder / TargetEncoder fit **ONLY** on the Train set.
- **Numerical Scaling:** StandardScaler / RobustScaler fit **ONLY** on the Train set.
- Use `sklearn.pipeline.Pipeline` or `ColumnTransformer` to enforce clean transformation bounds.

### Section 10: Model Selection & Hyperparameter Tuning
Compare at least 3 distinct model classes:
1. **Baseline / Benchmark:** `sklearn.dummy.DummyRegressor(strategy='mean')`
2. **Linear Baseline:** `RidgeRegression` or `Lasso`
3. **Tree-Based Ensembles:** `RandomForestRegressor`, `XGBRegressor`, or `LGBMRegressor`
- **Validation:** Perform TimeSeriesSplit or temporal Validation Split on the Train set.
- **Metric Selection:**
  - **RMSE (Root Mean Squared Error):** Primary metric to penalize large peak-demand prediction errors.
  - **MAE (Mean Absolute Error):** Secondary metric for overall average drift.
  - **$R^2$ Score:** Explained variance.

### Section 11 & 12: Model Evaluation & Justification
- Evaluate final models on the unseen Test set (October 2024).
- Generate comparison table: `Model | Train RMSE | Val RMSE | Test RMSE | Test MAE | Test R2`.
- Produce Feature Importance / SHAP summary plots.
- Complete the Section 12 justification matrix verifying stability, complexity trade-offs, and lack of leakage.

### Section 13: Project Summary & Business Handover
- State final model performance metrics against the naive baseline.
- Document limitations (e.g., lack of real-time weather data, non-continuous 52-week data series).
- Provide actionable recommendations for taxi fleet operators.

---

## 4. Final Notebook Output Checklist for Antigravity

Ensure the resulting Jupyter Notebook contains:
- [x] All 13 Markdown sections structured according to the course rubric.
- [x] Executed code cells with visible outputs (no missing runtime execution cells).
- [x] Summary tables with explicit row counts at each data transformation phase.
- [x] Clean, well-commented Python code.