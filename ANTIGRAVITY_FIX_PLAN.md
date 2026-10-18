# Project Remediation Plan for Antigravity

**Project:** NYC Yellow Taxi Hourly Zone Demand Forecasting  
**Objective:** Align the notebook and documentation with the actual data, target definition, model evaluation, and supported conclusions; make the work reproducible.

## Current Status

The notebook contains an end-to-end pipeline, from data loading through model comparison, and includes saved execution outputs. However, the current results do not support the claim that the model predicts demand for every zone-hour or is ready for production dispatch. The main issues are that aggregation drops hours with no trips and evaluation covers only one sampled seven-day period per season.

## Required Changes by Priority

### P0 - Fix Before Relying on the Results

#### 1. Build a complete zone-hour grid and include zero-demand hours

**Problem:** `groupby(...).size()` only creates rows for zone-hour pairs with at least one trip. The target therefore represents counts only for observed events, not demand for every Yellow Zone in every hour as described in the project.

**Required work:**

- After aggregation, create a complete sequence of hours for each seasonal block.
- Create the Cartesian product of those hours and the Yellow Zones in the defined study scope, then left-join the aggregated trip counts.
- Fill missing counts with `pickup_count = 0`.
- Exclude hours outside the four sampled blocks so that unloaded periods are not introduced.
- Explicitly state whether the study includes every Yellow Zone in the lookup table or only zones observed in the sample. For an all-zone demand forecast, use every Yellow Zone in the lookup table.
- Verify that each block contains 168 hours and that its row count equals `number of zones x number of hours`.
- Recompute EDA, target statistics, train/validation/test counts, and all model results after changing the target definition.

**Acceptance criteria:** Show the number of zones, hours, rows before and after zero filling, and zero-count rows. Assert that `PULocationID + pickup_hour` is unique and that the zone-hour grid has no gaps.

#### 2. Define the forecast task and prediction time precisely

**Problem:** The project proposes forecasts one to three hours ahead, but the experiment does not define a forecast origin or establish that every input feature is available before the target hour.

**Required work:**

- State whether the target is the current hour or a future hour such as `t+1` or `t+3`.
- For a future horizon, align the target to the future hour and use only features known at forecast time, such as calendar fields and zone identity.
- Document which features are known in advance. Do not use pickup information from after the forecast origin.
- Evaluate horizons separately, or select one horizon and use it consistently throughout the report and business recommendations.
- Do not claim the model supports advance fleet rebalancing until that horizon has been tested.

**Acceptance criteria:** Include an example with forecast origin, target timestamp, and input features. Assert that no feature timestamp is later than the forecast origin.

#### 3. Expand temporal evaluation beyond one week

**Problem:** Validation uses one seven-day Summer block and testing uses one seven-day Fall block. Results may depend on the particular weeks selected and do not establish performance on other weeks, holidays, or special events.

**Required work:**

- The Fall block may remain a holdout, but clearly state that it covers only seven sampled days.
- If the available data allows it, add multiple rolling temporal validation folds and report each fold along with the mean and standard deviation.
- If more data cannot be added, state this limitation prominently in the summary and avoid broad claims of stability or generalization.
- Show the date range and row count for every fold.

**Acceptance criteria:** Do not randomly split rows. Each validation fold must train on earlier data and evaluate on later data.

#### 4. Regenerate all outputs and conclusions from the revised notebook

Adding zero-demand rows or changing the forecast horizon invalidates the existing metrics and plots. Run the notebook from top to bottom and replace every affected result.

**Acceptance criteria:** All intended code cells have execution counts and current outputs. Row counts and values in the text, tables, plots, and conclusions agree; no stale results remain.

### P1 - Improve the Experiment and Interpretation

#### 5. Fit preprocessing only on each validation fold's training data

**Required work:**

- Use `Pipeline` and `ColumnTransformer` inside each fold, fitting preprocessing only on that fold's training data.
- Verify that validation and test data are never used to fit scalers, encoders, or feature-selection steps.
- After selecting the model and parameters using validation, fit the final model on all training data and evaluate the test set once.

#### 6. Use a section title that matches the work performed

The current notebook sets model parameters in advance and compares models; it does not perform a systematic parameter search. Choose one of these options:

- Rename the section to **Model Comparison** if no tuning is performed, or
- Add a parameter search using temporal folds on training data only. Do not select models based on test results.

#### 7. Add a time-aware baseline suited to forecasting

Keep `DummyRegressor` as a reference baseline and add at least one meaningful temporal baseline, such as the previous matching hour/day or a historical average, provided it can be computed without looking into the future. Document the information each baseline uses.

#### 8. Analyze errors in groups relevant to deployment

- Report MAE and RMSE by time period, zone, and demand level (for example, low, medium, and high), with sample counts.
- Report prediction bias, especially the Evening Peak underprediction visible in the current results, where mean actual demand is notably higher than mean predicted demand.
- For a count target, consider adding an interpretable count metric such as Poisson deviance, with its assumptions explained.
- Do not label an error level "acceptable" without an operational threshold or stakeholder input.

#### 9. Keep business conclusions within the evidence

- Remove or revise claims such as **Production-Ready**, "real-time dispatch," and a 20-30% reduction in deadheading unless directly supported by an experiment.
- Report the latest metrics with their date range, units, and comparison baseline.
- Present fleet-dispatch recommendations as proposals for further evaluation, not proven outcomes.
- State the limitations: four sampled seven-day blocks, no holiday or special-event coverage, and no weather or other transport-service data.
- Do not leave hard-coded metrics in Markdown if they no longer match the latest results.

### P2 - Improve Reproducibility and Handoff Quality

#### 10. Add project run instructions to `README.md`

Include:

- Project purpose and file structure.
- Dependency installation instructions or a `requirements.txt` with important package versions.
- Instructions for opening and running the notebook in order.
- Data sources and download steps when files are absent from `data/`.
- Approximate RAM and storage needs, plus common issues.
- The environment used to generate notebook outputs and how to reproduce them.

#### 11. Replace team placeholders and verify submission details

- Replace `Member 2`-`Member 5` and `XXX` student IDs with accurate details, or remove placeholders before submission.
- Verify team name, work percentages, and deadlines.
- Check that Thai text renders correctly in the notebook and README in Jupyter and Colab.

#### 12. Fix minor output inconsistencies

- Verify the inference benchmark row count. The current text says 8,393 while the recorded test set contains 8,344 rows. Calculate the count from `len(X_test)` and use the resulting value.
- Remove or resolve plotting warnings such as the current FutureWarning where practical.
- Select the champion model from the latest evaluation results rather than hard-coding LightGBM as the winner.

## Recommended Work Order

1. Fix the target definition and build the zero-filled zone-hour grid.
2. Decide on and implement the forecast horizon.
3. Update temporal splits and make preprocessing leakage-safe.
4. Rerun model comparison and error analysis.
5. Revise tables, plots, conclusions, and business recommendations to match the new results.
6. Complete the README and remove placeholders.
7. Run the notebook from top to bottom in the documented environment and confirm that all outputs are present and no cells fail.

## Delivery Requirements

- A fully executed notebook with outputs matching the latest source code.
- Row-count summaries for every data transformation, including the number of zero-demand rows.
- A clearly defined comparison table with train, validation, and test metrics for baselines and candidate models.
- Evidence of chronological splitting and preprocessing fit only on training data.
- Conclusions that stay within the scope of the data and experiments.
- A README that enables another user to reproduce the work.

> **Implementation note:** Do not retain old metrics such as RMSE, MAE, R^2, or row counts after changing the data or target without recomputing them. Values mentioned in this review identify issues; they are not target results to preserve.
