# E-commerce Behavior Analysis and Purchase Prediction

An end-to-end machine-learning project that predicts an e-commerce purchase outcomes from session-level behavioral data. The project focuses on **extreme class imbalance, leakage-safe preprocessing, robust model evaluation, threshold analysis, sensitivity testing, and model interpretability.**

| Project Overview    | Details                                                  |
| ------------------- | -------------------------------------------------------- |
| Dataset             | 8,000 simulated e-commerce sessions                      |
| Usable observations | 7,840 sessions after removing rows with missing targets  |
| Target              | Purchase: Buyer (1) / Non-buyer (0)                      |
| Models              | Logistic Regression, Decision Tree, Random Forest        |
| Selected model      | Logistic Regression                                      | 
| Validation          | Repeated stratified 5-fold cross-validation × 20 repeats |
| Main challenge      | Only 13 non-buyers in the usable dataset                 |

**Key takeaway:** Logistic Regression achieved the strongest minority-class performance among the evaluated models. However, the extremely small number of original non-buyers limits the reliability and generalizability of the estimates.

---

## Table of Contents
- Project Overview
- Objectives
- Dataset
- Tools and Technologies
- Methodology
- Models and Evaluation Metrics
- Results
- Threshold Analysis
- Sensitivity Analysis
- Model Interpretability
- PCA Exploratory Analysis
- Business Recommendations
- Limitations
- Project Outputs
- Reproducibility
- Loading the Saved Model
- Conclusion

---

## Project Overview

E-commerce platforms generate behavioral data that can help explain purchasing patterns and support data-driven decisions. This project develops a predictive workflow to determine whether a browsing session ends in a purchase, with particular emphasis on identifying the rare sessions that do not convert.

The main analytical question is:
**To what extent can session-level behavioral data predict purchase outcomes when non-buying sessions are extremely rare?**

The project goes beyond a basic classification exercise by combining data-quality assessment, leakage-safe preprocessing, imbalance-handling strategies, repeated cross-validation, out-of-fold evaluation, threshold analysis, model interpretation, and exploratory dimensionality reduction.

Because the dataset contains only 13 non-buyers, accuracy alone is insufficient for evaluating model quality. Model selection therefore prioritizes minority-class PR-AUC, recall, precision, and Matthews Correlation Coefficient (MCC).

**Important:** The dataset is simulated. The project demonstrates an analytical methodology rather than establishing validated performance on real e-commerce customers.

---

## Objectives

- Assess data quality, missing values, duplicates, and potential outliers.
- Prepare numerical, binary, and categorical features for machine learning.
- Prevent data leakage during preprocessing and resampling.
- Address extreme class imbalance using SMOTE, SMOTENC, and class weighting.
- Compare Logistic Regression, Decision Tree, and Random Forest.
- Evaluate minority-class performance using imbalance-aware metrics.
- Select the most appropriate model for rare non-buyer detection.
- Analyze precision–recall trade-offs across classification thresholds.
- Test the sensitivity of results to alternative imbalance-handling strategies.
- Interpret model coefficients and tree-based feature importance.
- Explore feature structure using PCA.
- Save the final fitted pipeline and reproducible analytical outputs.

---

## Dataset 

The project uses simulated session-level e-commerce data.

### Data quality summary

| Item                               | Value |
| ---------------------------------- | ----- |
| Raw observations                   | 8,000 |
| Raw columns                        | 14    |
| Missing values per original column | 160   |
| Duplicate rows removed             | 0     |
| Rows removed due to missing target | 160   |
| Final modelling observations       | 7,840 |
| Buyers                             | 7,827 |
| Non-buyers                         | 13    |
| Non-buyer prevalence               | 0.17% |

Rows with missing `purchase` values are removed rather than assigning inferred labels. Missing predictor values are handled within the modelling pipeline.

The `user_id` identifier is removed because it is not used as a behavioral predictor. Potential IQR outliers are reported but retained, avoiding unnecessary information loss in an already extremely small minority class.

### Target variable

The target is `purchase`:

| Value | Meaning                                       |
| ----- | --------------------------------------------- |
| 1     | Buyer / session ended in a purchase           |
| 0     | Non-buyer / session did not end in a purchase |

### Predictor variables

| Feature              | Type        | Description                          |
|----------------------|-------------|--------------------------------------|
| `age`                | Continuous  | User age                             |
| `time_on_site`       | Continuous  | Time spent during the session        |
| `pages_viewed`       | Continuous  | Number of pages viewed               |
| `previous_purchases` | Continuous  | Number of previous purchases         |
| `cart_items`         | Continuous  | Number of items added to the cart    |
| `avg_session_time`   | Continuous  | Average session duration             |
| `bounce_rate`        | Continuous  | Session bounce rate                  |
| `discount_seen`      | Binary      | Whether a discount was displayed     |
| `ad_clicked`         | Binary      | Whether an advertisement was clicked |
| `returning_user`     | Binary      | Whether the user is returning        |
| `gender`             | Categorical | User gender                          |
| `device_type`        | Categorical | Device used during the session       |

### The class imbalance challenge

Only 13 of the 7,840 usable sessions belong to the non-buyer class, representing approximately 0.17% of the dataset.

A classifier that predicts every session as a buyer would achieve approximately 99.83% accuracy while failing to identify a single non-buyer.

For this reason, accuracy and weighted F1 are reported for context but are not treated as the main criteria for model selection.

---

## Tools and Technologies

| Technology       | Purpose                                                           |
| ---------------- | ----------------------------------------------------------------- |
| Python           | Main programming language                                         |
| Pandas           | Data cleaning, transformation, and analysis                       |
| NumPy            | Numerical operations                                              |
| scikit-learn     | Preprocessing, classification, cross-validation, metrics, and PCA |
| imbalanced-learn | SMOTE, SMOTENC, and leakage-safe resampling pipelines             |
| Matplotlib       | Plot generation and export                                        |
| Seaborn          | Statistical visualisations                                        |
| Joblib           | Saving and loading the fitted model                               |
| Pathlib / OS     | File paths, environment variables, and project configuration      |

---

## Methodology

### 1. Data preparation and quality assessment

The workflow:
1. Loads and inspects the raw CSV dataset.
2. Examines missing values and dataset dimensions.
3. Removes `user_id`.
4. Checks and removes duplicate records.
5. Removes observations with missing target labels.
6. Separates predictors from the target.
7. Reports IQR-based outliers without automatically removing them.
8. Saves data-quality summaries and exploratory figures.

### 2. Leakage-safe preprocessing

All learned preprocessing operations are placed inside the modelling pipeline and fitted using only the training portion of each cross-validation fold.

| Feature type          | Processing                                          |
| --------------------- | --------------------------------------------------- |
| Continuous variables  | Median imputation and standard scaling              |
| Binary indicators     | Most-frequent-value imputation                      |
| Categorical variables | Most-frequent-value imputation and one-hot encoding |

Categorical encoding uses fixed categories with ```python drop="first"```, using Female as the gender reference and Desktop as the device reference.

This approach ensures that preprocessing statistics and transformations are learned from training data rather than from the validation fold.

### 3. Handling class imbalance

The main modelling configuration combines **SMOTE with balanced class weights.**

SMOTE generates synthetic minority-class examples within each training fold, targeting a minority-to-majority ratio of `0.4`. The model also assigns balanced class weights to give greater importance to the underrepresented class.

The project compares this approach against alternative configurations in the sensitivity analysis.

### 4. Repeated stratified cross-validation

The main model comparison uses:
- 5 stratified folds
- 20 repeats
- Random seed: `42`

This gives **100 train/validation evaluations per model configuration.**

Stratification helps preserve the class distribution across folds, while repetition reduces dependence on a single fold assignment. However, repeated cross-validation does not create additional independent non-buyer observations; the dataset still contains only 13 original non-buyers.

### 5. Out-of-fold evaluation and threshold analysis

Repeated out-of-fold predictions are used to examine:
- Precision-recall curves for the non-buyer class
- Precision and recall at different probability thresholds
- Average confusion-matrix outcomes
- The relationship between alert volume and non-buyer detection

Each out-of-fold prediction is produced by a model that was not trained on that observation.

### 6. Model interpretation and exploratory analysis

The project generates Logistic Regression coefficient summaries, Decision Tree visualizations, tree-based feature importance, and PCA visualizations. PCA is used for exploratory analysis and is not part of the predictive model evaluation.

---

## Models and Evaluation Metrics

### Models

- **Logistic Regression:** Interpretable linear classifier and selected final model.
- **Decision Tree:** Tree-based classifier for identifying split-based decision patterns.
- **Random Forest:** Ensemble classifier for capturing potentially nonlinear relationships.

### Evaluation metrics

| Metric                     | Purpose                                                                      |
| -------------------------- | ---------------------------------------------------------------------------- |
| ROC-AUC                    | Evaluates how well predicted scores rank the two classes                     | 
| PR-AUC / Average Precision | Evaluates precision–recall ranking for the rare non-buyer class              | 
| Minority recall            | Measures the proportion of actual non-buyers detected                        | 
| Minority precision         | Measures the proportion of flagged sessions that are actual non-buyers       | 
| MCC                        | Summarizes classification quality using all four confusion-matrix components | 
| Accuracy                   | Measures overall classification correctness                                  | 
| Weighted F1                | Combines precision and recall, weighted by class support                     | 

The minority class is treated as the positive class for PR-AUC, recall, and precision calculations. This makes the evaluation directly relevant to identifying non-buying sessions.

---

## Results

The following table summarizes the main configuration: SMOTE + class weighting.

Values are reported as mean ± standard deviation across the 100 repeated cross-validation evaluations.

| Model               | ROC-AUC           | PR-AUC            | MCC               | Weighted F1   | Accuracy      | Recall (No buy)   | Precision (No buy) |
| ------------------- | ----------------- | ----------------- | ----------------- | ------------- | ------------- | ----------------- | ------------------ |
| Logistic Regression | **0.999 ± 0.001** | **0.783 ± 0.184** | **0.512 ± 0.103** | 0.997 ± 0.001 | 0.996 ± 0.001 | **0.947 ± 0.135** | 0.286 ± 0.103      |
| Decision Tree       | 0.752 ± 0.158     | 0.115 ± 0.104     | 0.230 ± 0.177     | 0.997 ± 0.001 | 0.996 ± 0.002 | 0.367 ± 0.294     | 0.163 ± 0.170      |
| Random Forest       | 0.991 ± 0.026     | 0.498 ± 0.248     | 0.314 ± 0.317     | 0.998 ± 0.001 | 0.999 ± 0.001 | 0.242 ± 0.258     | **0.432 ± 0.452**  |

### Selected model: Logistic Regression

Logistic Regression was selected because it achieved the strongest overall results for the project's main objective: detecting rare non-buying sessions.

Key findings:

- **ROC-AUC of 0.999:** exceptionally strong ranking performance in the repeated validation results.
- **PR-AUC of 0.783:** substantially higher than Random Forest (0.498) and Decision Tree (0.115).
- **Minority recall of 0.947:** identifies approximately 94.7% of non-buyers on average.
- **MCC of 0.512:** stronger overall classification association than the alternatives.
- **Comparatively low variability:** more stable results on several key metrics than the tree-based models.

Its non-buyer precision of 0.286 means that approximately 28.6% of sessions flagged as non-buyers are are genuine non-buyers on average. Although moderate in absolute terms, this is substantially higher than the original non-buyer prevalence of approximately 0.17%.

### Decision Tree

The Decision Tree was the weakest model overall. Its minority recall of 0.367 means it identifies approximately 36.7% of non-buyers, while its minority precision of 0.163 is lower than Logistic Regression's.

Its high variability across folds is consistent with the sensitivity of a single tree to changes in a very small minority sample. Its performance is therefore less suitable for a recall-focused non-conversion screening task.

### Random Forest

Random Forest achieved the highest overall accuracy (0.999) and the highest mean minority precision (0.432). However, its minority recall was only 0.242, meaning that it detected approximately 24.2% of non-buyers on average.

Its high accuracy largely reflects the dominant buyer class, while its minority recall indicates that most actual non-buyers are missed. Its precision and PR-AUC also show substantial variability across folds.

### Overall interpretation

Logistic Regression is the preferred model for the stated objective, despite not achieving the highest accuracy. The results illustrate why rare-class ranking, recall, precision, and MCC are more informative than accuracy alone when the target is extremely imbalanced.

---

## Threshold Analysis

The selected Logistic Regression model was evaluated across different decision thresholds using repeated out-of-fold probabilities.

A session is flagged as a likely non-buyer when:

`P(No Buy) ≥ threshold`

| Threshold | Recall | Precision | Sessions flagged per pass | Non-buyers caught per pass |
| --------- | ------ | --------- | ------------------------- | -------------------------- |
| 0.05      | 1.000  | 0.108     | 120.4                     | 13.0                       |
| 0.10      | 1.000  | 0.135     | 96.4                      | 13.0                       |
| 0.20      | 1.000  | 0.178     | 73.4                      | 13.0                       |
| 0.30      | 1.000  | 0.214     | 60.8                      | 13.0                       |
| 0.40      | 1.000  | 0.245     | 53.2                      | 13.0                       |
| 0.50      | 0.969  | 0.277     | 45.6                      | 12.6                       |
| 0.60      | 0.923  | 0.315     | 38.2                      | 12.0                       |
| 0.70      | 0.892  | 0.347     | 33.4                      | 11.6                       |
| 0.80      | 0.877  | 0.410     | 27.8                      | 11.4                       |
| 0.90      | 0.800  | 0.487     | 21.4                      | 10.4                       |
| 0.95      | 0.708  | 0.567     | 16.4                      | 9.2                        |

*Counts are average outcomes per full out-of-fold evaluation pass, not counts from a single independent test set.*

### Default threshold: 0.50

At the default threshold, the average confusion matrix reports:

| Outcome                                    | Average sessions per full pass |
| ------------------------------------------ | ------------------------------ |
| Non-buyers correctly flagged               | 12.6                           |
| Non-buyers incorrectly predicted as buyers | 0.4                            |
| Buyers incorrectly flagged as non-buyers   | 33.0                           |
| Buyers correctly predicted as buyers       | 7,794.0                        |

The threshold analysis demonstrates a clear trade-off:

- **Lower thresholds** catch more non-buyers but flag more sessions.
- **Higher thresholds** improve precision and reduce alert volume but miss more non-buyers.

The appropriate threshold depends on the cost of false-positive interventions versus the cost of missing a non-buyer. For example, a low-cost support prompt may justify a lower threshold, whereas an expensive incentive may require a higher one.

---

## Sensitivity Analysis

Four imbalance-handling strategies were evaluated:

1. SMOTE + class weighting
2. SMOTE only
3. SMOTENC + class weighting
4. Class weighting only, without resampling

The analysis examines PR-AUC, MCC, recall, and precision to determine whether model performance changes with the imbalance-handling strategy.

### Main findings

- **Logistic Regression remained the strongest model for minority-class ranking** across the tested configurations.
- SMOTE only achieves a PR-AUC of 0.780 and MCC of 0.539 for Logistic Regression, compared with 0.783 and 0.512 under the main configuration.
- SMOTENC + class weighting produces a lower Logistic Regression PR-AUC (0.711) but higher mean precision (0.365) and lower recall (0.828) than the main configuration.
- Class weighting alone produces very high Logistic Regression recall, but its precision and overall balance should be considered when choosing an operating threshold.

These findings show that imbalance handling affects the balance between detecting non-buyers and avoiding false-positive flags. No single configuration should be considered universally optimal without reference to the intended business use.

---

## Model Interpretability

### Logistic Regression coefficients

Coefficient analysis helps examine the direction and relative strength of predictive associations.

In the fitted model, the most prominent purchase-associated signals included:

- **`cart_items`:** strong positive association with purchase.
- **`previous_purchases`:** strong positive association with purchase.
- **`discount_seen`:** positive association in the simulated dataset.
- **`avg_session_time` and `time_on_site`:** positive associations with purchase.
- **`returning_user`:** positive association with purchase.
- **`bounce_rate`:** strong negative association with purchase.

Continuous features are standardized, so their coefficients describe changes in purchase log-odds associated with an approximately one-standard-deviation increase. Binary features are not standardized, so their coefficient magnitudes are not directly comparable with those of continuous features.

These coefficients represent predictive associations, not causal effects. Demographic variables should not be used to justify differential treatment without appropriate fairness, privacy, and legal review.

### Tree-based feature importance

Decision Tree and Random Forest importance estimates also highlight:

- `cart_items`
- `previous_purchases`
- `avg_session_time`
- `bounce_rate`

The models broadly indicate that cart activity, purchase history, and session engagement carry useful predictive information in this dataset. Tree-based importance measures reflect contributions to the fitted models; they do not establish that changing a feature will cause a purchase.

---

## PCA Exploratory Analysis

Principal Component Analysis was applied to the real, unresampled sessions after preprocessing and standardization. PCA was used for exploratory analysis, not as part of predictive model selection.

### Variance explained

| Component              | Variance explained | Main interpretation                                             |
| ---------------------- | ------------------ | --------------------------------------------------------------- |
| PC1                    | 14.1%              | Session duration and engagement-time                            |
| PC2                    | 10.9%              | Device-type contrast                                            |
| PC3                    | 8.1%               | Mixed demographic and behavioural-history structure             |
| First three components | 33.1%              | Partial representation of total feature variance representation |

The first three components explain 33.1% of the total feature variance.

The exploratory projections show substantial overlap between buyers and non-buyers rather than a completely isolated non-buyer cluster. This supports the use of supervised probabilistic classification rather than expecting the rare class to form a clearly separated group in a low-dimensional projection.

PCA is unsupervised: it does not use the purchase target to calculate components, and explained variance is not a measure of classification performance.

---

## Business Recommendations

The results suggest several possible applications, subject to validation with real-world data.

### 1. Use predictions for prioritization
Logistic Regression could be evaluated as a screening model to prioritize sessions for low-cost interventions, such as checkout assistance, relevant product recommendations, or non-intrusive support prompts.

### 2. Select the threshold based on business value
The threshold should reflect he cost of unnecessary interventions, the value of a successful conversion, and the consequences of missing a non-buyer. Threshold selection should be revisited using independent real-world data.

### 3. Investigate important behavioral signals
Cart activity, purchase history, session duration, and bounce behavior can guide further analysis and hypothesis generation. These associations should not be interpreted as causal findings.

### 4. Test interventions experimentally
The model predicts which sessions may not convert; it does not prove which action will change the outcome. A/B testing can evaluate whether interventions such as checkout support, free-shipping messages, or personalized recommendations generate incremental conversion or profit.

### 5. Validate and monitor before deployment
Before real-world use, evaluate the model on a larger, representative dataset and monitor minority-class PR-AUC, recall, precision, alert volume, probability calibration, feature distributions, and business outcomes over time.

---

## Limitations

- **Extremely small minority class:** only 13 original non-buyers are available. Fold-level estimates are sensitive to individual observations.
- **Simulated data:** the dataset may not represent real customer behavior, commercial conditions, or production data quality.
- **Synthetic oversampling:** SMOTE and SMOTENC generate examples from a very limited number of original non-buyers; these synthetic observations cannot replace real minority-class data.
- **No independent external validation:** repeated cross-validation provides internal evaluation, but the final model still requires validation on new, representative data.
- **Probability calibration:** resampling and class weighting can affect how model scores correspond to real-world probabilities. Calibration should be assessed independently before interpreting scores as actual purchase or non-purchase probabilities.
- **No causal inference:** model coefficients, feature importance, and PCA reveal associations or structure, not proof of cause and effect.
- **Exploratory PCA:** the first three components explain only 33.1% of the feature variance and do not replace supervised evaluation.

Consequently, the results should be interpreted as a methodological demonstration of imbalanced classification rather than a guarantee of production performance.

---

## Project Outputs

The pipeline generates three main output folders: `figures/`, `results/`, and `cv_cache/`.

### Figures

| File                            | Description                                                                |
| ------------------------------- | -------------------------------------------------------------------------- |
| `01_missing_values.png`         | Missing values by original data column                                     |
| `02_class_imbalance.png`        | Buyer and non-buyer distribution                                           |
| `03_feature_profiles.png`       | Continuous-feature profiles by purchase outcome                            |
| `04_cv_model_comparison.png`    | Repeated cross-validation model comparison                                 |
| `05_pr_threshold_confusion.png` | Precision-recall curves, threshold trade-off, and average confusion matrix |
| `06_ablation.png`               | Sensitivity analysis across imbalance strategies                           |
| `07_lr_coefficients.png`        | Logistic Regression coefficients interpretation                            |
| `08_decision_tree.png`          | Final Decision Tree visualization                                          |
| `09_feature_importance.png`     | Tree-based feature importance                                              |
| `10_pca_overview.png`           | PCA variance, loadings, and projections                                    |
| `11_pca_3d.png`                 | Three-dimensional PCA visualisation                                        |

### Result tables and model

| File                                         | Description                                                          |
| -------------------------------------------- | -------------------------------------------------------------------- |
| `data_quality_summary.csv`                   | Dataset quality summary                                              |
| `cv_results_summary.csv`                     | Main Cross-validation results as mean ± standard deviation           |
| `cv_results_means.csv`                       | Numeric cross-validation means                                       |
| `paired_comparison.csv`                      | paired comparison between Logistic Regression and alternative models |
| `threshold_analysis_logistic_regression.csv` | Threshold-performance results                                        |
| `cv_results_smote_only.csv`                  | SMOTE-only results                                                   |
| `cv_results_smotenc.csv`                     | SMOTENC + class-weight results                                       |
| `cv_results_weight_only.csv`                 | Class-weight-only results                                            |
| `ablation_comparison.csv`                    | Combined sensitivity-analysis results                                |
| `lr_coefficients.csv`                        | Logistic Regression coefficients                                     |
| `tree_feature_importance.csv`                | Tree-based feature importance                                        |
| `pca_loadings.csv`                           | PCA loadings                                                         |
| `final_logistic_regression_pipeline.joblib`  | Saved final Logistic Regression pipeline                             |

The `cv_cache/` folder stores cached evaluation results to reduce rerun time. Delete this cache after changing the dataset, feature engineering, preprocessing, scoring, or model settings that affect evaluation results.

---

## Reproducibility

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd E-commerce-Behavior-Analysis-Purchase-Prediction
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

### 4. Configure the dataset path

The script supports the `ECOM_DATA` environment variable.

**Windows PowerShell**

```powershell
$env:ECOM_DATA = "C:\path\to\ecommerce_user_behavior_8000.csv"
```

**macOS / Linux**

```bash
export ECOM_DATA="/path/to/ecommerce_user_behavior_8000.csv"
```
If `ECOM_DATA` is not set, update the default path in the script to match you local dataset location.

### 5. Configure runtime settings (optional)

The script provides configurable environment variables.

| Environment Variable | Default               | Purpose                                                 |
| ---------------------| --------------------- | ------------------------------------------------------- |
| `ECOM_DATA`          | Configured local path | CSV input file path                                     |
| `ECOM_REPEATS`       | `20`                  | Number of repeated CV rounds                            |
| `ECOM_OOF_REPEATS`   | `5`                   | Number of OOF repeats for curves and threshold analysis |

For a quicker test run:

**Windows PowerShell**

```powershell
$env:ECOM_REPEATS = "2"
$env:ECOM_OOF_REPEATS = "1"
```

**macOS / Linux**

```bash
export ECOM_REPEATS=2
export ECOM_OOF_REPEATS=1
```

Use the default settings for the full evaluation.

### 6. Run the pipeline

```bash
python ecommerce_pipeline.py
```

If the script is stored in a `Data` directory, run it using its actual location, for example

```powershell
py "Data\ecommerce_pipeline.py"
```

The pipeline cleans the data, runs model comparisons and sensitivity analysis, generates figures and result tables, fits the final Logistic Regression pipeline, and performs PCA exploration.

---

## Loading the Saved Model

After running the pipeline, the fitted model is saved to:

`results/final_logistic_regression_pipeline.joblib`

Example:

```python
import joblib
import pandas as pd

pipeline = joblib.load("results/final_logistic_regression_pipeline.joblib")

new_sessions = pd.DataFrame({
    "age": [30],
    "gender": ["Female"],
    "device_type": ["Mobile"],
    "time_on_site": [180],
    "pages_viewed": [8],
    "previous_purchases": [2],
    "cart_items": [1],
    "discount_seen": [1],
    "ad_clicked": [0],
    "returning_user": [1],
    "avg_session_time": [120],
    "bounce_rate": [0.25]
})

prediction = pipeline.predict(new_sessions)
probabilities = pipeline.predict_proba(new_sessions)

print("Predicted class:", predictions)
print("Class order:", pipeline.named_steps["clf"].classes_)
print("Class probabilities:", probabilities)
```

The class order is available from the fitted classifier's classes_ attribute. Always check this order before interpreting the columns returned by predict_proba().

---

## Conclusion

This project demonstrates an end-to-end approach to purchase prediction under extreme class imbalance, combining data preparation, leakage-safe preprocessing, SMOTE, class weighting, repeated stratified cross-validation, threshold analysis, sensitivity testing, and model interpretation.

Under the main configuration, Logistic Regression achieved a mean PR-AUC of 0.783, non-buyer recall of 0.947, and MCC of 0.512, outperforming the tested Decision Tree and Random Forest models on the principal minority-class objectives.

Its results support its selection as the preferred model for further investigation as a non-conversion screening tool. However, the small number of original non-buyers and the simulated nature of the dataset mean that the findings require independent validation before any real-world deployment.

**The central lesson:** a useful predictive model should be judged not only by overall accuracy, but also by which cases it detects, which errors it makes, how stable its performance is, and whether its predictions support an appropriate business decision.

---

