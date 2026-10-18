# NYC Yellow Taxi Hourly Zone Demand Forecasting (Big Data Analytics)

**Course:** Big Data Analytics — Part 2: Data Management and Machine Learning Pipeline  
**Dataset:** Official NYC Taxi & Limousine Commission (TLC) 2024 Trip Record Data  
**Task Type:** Time-Series / Tabular Supervised Regression  
**Target Variable:** `pickup_count` (Hourly pickup trips per Yellow Zone, with complete zero-demand imputation)  
**Study Scope:** 55 Yellow Zones in Manhattan, NYC  

---

## 1. Project Overview & Business Motivation

Yellow Taxis in New York City face severe supply-demand imbalances:
- **Undersupply during peak hours:** Passengers face long wait times, and taxi operators miss commercial revenue.
- **Deadheading / Cruising during off-peak hours:** Taxis roam empty searching for fares, causing traffic congestion, excessive fuel consumption, and unnecessary emissions.

**Core Research Question:**  
*Can we accurately forecast the hourly taxi pickup demand (`pickup_count`) for all 55 Yellow Zones in Manhattan 1 hour in advance ($H=1$) using temporal, cyclical, seasonal, and spatial indicators without future data leakage?*

---

## 2. Repository Structure

```
proj2bigdata/
├── MiniProject_Part2_BigData_ML.ipynb   # Fully executed main Jupyter Notebook (all 28 cells)
├── build_notebook.py                    # Script to regenerate notebook structure
├── requirements.txt                     # Python dependencies
├── taxi_zone_lookup.csv                 # Official TLC Taxi Zone Lookup metadata
├── handoff_project_specification.md     # Initial project technical specification
├── ANTIGRAVITY_FIX_PLAN.md              # Rigorous remediation & quality plan
├── README.md                            # Comprehensive run and reproducibility guide
└── data/                                # Parquet datasets (ignored in git)
    ├── yellow_tripdata_2024-01.parquet
    ├── yellow_tripdata_2024-04.parquet
    ├── yellow_tripdata_2024-07.parquet
    └── yellow_tripdata_2024-10.parquet
```

---

## 3. Dataset & Data Engineering Details

- **Raw Data Volume:** 13,389,587 raw trips across 4 months of 2024.
- **Seasonal Sampling:** 7-day continuous blocks (8th–14th) for Winter (Jan), Spring (Apr), Summer (Jul), and Fall (Oct) = 3,126,735 raw trips (satisfying the course threshold of > 1,000,000 rows).
- **Service Zone Scope:** Filtered strictly for `PU_service_zone == 'Yellow Zone'` = 2,706,039 trips.
- **Complete Cartesian Grid:**  
  $$\text{55 Zones} \times 168 \text{ Hours} \times 4 \text{ Seasons} = \mathbf{36,960 \text{ Zone-Hour Rows}}$$
  - **Positive Demand Hours (> 0):** 33,129 rows (89.63%)
  - **Zero Demand Hours (= 0):** 3,831 rows (10.37%) imputed with 0 to eliminate zero-truncation selection bias.

---

## 4. Modeling & Validation Architecture

- **Forecast Horizon:** $H=1$ hour ahead ($T_{origin} = t - 1\text{ hour}$).
- **Leakage Protection:** All input features (`hour`, `day_of_week`, `is_weekend`, `month`, `season`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `PULocationID`) are deterministic calendar/spatial fields known strictly prior to the target hour.
- **Cross-Validation Strategy:**
  - Expanding-Window Temporal Validation on Training Data (Seasons 1–3: 27,720 rows):
    - **Fold 1:** Train on Winter (9,240 rows) $\rightarrow$ Validate on Spring (9,240 rows)
    - **Fold 2:** Train on Winter + Spring (18,480 rows) $\rightarrow$ Validate on Summer (9,240 rows)
  - Preprocessing (`ColumnTransformer` with `OneHotEncoder` and `StandardScaler`) is **fit ONLY on the training fold**.
- **Holdout Test Set:** Fall / October 2024 (9,240 rows, 168 hours) evaluated strictly once.

---

## 5. Summary of Results (Holdout Test Set: Fall 2024)

| Model Architecture | Model Name | Train RMSE | Val Mean RMSE | Test RMSE | Test MAE | Test $R^2$ | RMSE Reduction vs Baseline | Inference Speed |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline** | Dummy (Global Mean) | 87.48 | 90.25 | 99.94 | 71.30 | -0.0107 | 0.00% | - |
| **Historical Baseline** | Historical Zone-DOW-Hour Mean | 19.18 | 31.05 | 28.48 | 14.94 | 0.9179 | +71.50% | ~0.04 s |
| **Linear Model** | Ridge Regression ($L_2$) | 57.72 | 63.94 | 65.86 | 42.61 | 0.5611 | +34.10% | < 0.01 s |
| **Ensemble (Bagging)** | Random Forest | 40.17 | 48.20 | 52.11 | 35.34 | 0.7252 | +47.86% | ~0.75 s |
| **Ensemble (Boosting)** | XGBoost | 28.13 | 37.72 | 41.67 | 24.59 | 0.8243 | +58.31% | ~0.19 s |
| **Ensemble (Boosting)** | **LightGBM (Champion ML)** | **26.20** | **37.02** | **39.44** | **22.45** | **0.8426** | **+60.54%** | **~0.16 s** |

### Key Takeaways:
1. **LightGBM** achieved the best Machine Learning performance with **Test RMSE 39.44** and **$R^2 = 84.26\%$**, outperforming linear regression by over 28 percentage points in explained variance.
2. **Subgroup Error Analysis:** The model performs with low error during Low-Demand hours (MAE 3.65), but exhibits an underprediction bias during the Evening Peak (Mean Bias +10.9 trips/hour), requiring operational safety buffers in real-world deployment.

---

## 6. How to Reproduce & Run the Project

### Environment Requirements
- **OS:** Windows 10/11, macOS, or Linux
- **Python:** 3.10, 3.11, or 3.12 (64-bit recommended)
- **RAM:** Minimum 8 GB (16 GB recommended)
- **Disk Space:** ~500 MB for parquet files and notebook

### Step 1: Clone the Repository
```bash
git clone https://github.com/Thanaphon-673020253-2/proj2bigdata.git
cd proj2bigdata
git checkout suk
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Jupyter Notebook
Open the notebook in Jupyter Lab, Jupyter Notebook, or VS Code:
```bash
jupyter lab MiniProject_Part2_BigData_ML.ipynb
```
Select `Run All Cells`. If the raw `.parquet` files are not found in `data/`, Cell 5 will automatically download them from the official NYC TLC CloudFront servers.

To execute headlessly from the command line:
```bash
python -m nbconvert --to notebook --execute --inplace MiniProject_Part2_BigData_ML.ipynb
```

---

## 7. Project Limitations & Future Work

1. **Seasonal Sample Blocks:** Evaluated on 4 representative 7-day blocks (672 hours total) rather than continuous 365-day data.
2. **External Data Constraints:** Weather conditions (precipitation, snow) and subway delays (MTA) are not yet integrated into the feature pipeline.
3. **Future Extension:** Integrate NOAA Weather API and implement Spatio-Temporal Graph Neural Networks (GNN) to capture network traffic topology.