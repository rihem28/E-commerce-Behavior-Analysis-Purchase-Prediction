# E-commerce-Behavior-Analysis-Purchase-Prediction

## 1. Introduction

This project analyzes 8,000 e-commerce user sessions to understand and predict which sessions **do not** result in a purchase. What starts as a standard binary classification task turns into an exercise in rigorous imbalanced-data methodology once the true class distribution is revealed: **99.84% of sessions convert**, leaving only **13 non-buying sessions (0.16%)** in the entire dataset. The project's real contribution is not just a trained model, but a documented, defensible process for evaluating that model without being misled by the inflated accuracy and ROC-AUC scores that this level of imbalance produces almost automatically.

## 2. Dataset Description and the Core Issue

**Files:**
- Raw data: `ecommerce_user_behavior_8000.csv`
- Cleaned data (post-cleaning/encoding): `ecommerce_user_behavior_cleaned.csv`

**Features:**

| Feature | Type | Description |
|---|---|---|
| `age` | numeric | User age |
| `gender` | categorical → label-encoded | User gender |
| `device_type` | categorical → one-hot encoded | Desktop / Mobile / Tablet |
| `time_on_site` | numeric | Total time spent on site |
| `pages_viewed` | numeric | Number of pages viewed |
| `previous_purchases` | numeric | Count of prior purchases |
| `cart_items` | numeric | Items added to cart |
| `avg_session_time` | numeric | Average session duration |
| `bounce_rate` | numeric | Session bounce rate |
| `discount_seen` | binary | Whether a discount was displayed |
| `ad_clicked` | binary | Whether the user clicked an ad |
| `returning_user` | binary | Returning vs. new visitor |
| `purchase` | binary (target) | 1 = Buy, 0 = No Buy |

**The core issue — extreme class imbalance:**

| Class | Sessions | Share |
|---|---|---|
| Purchase = 1 (Buy) | 7,987 | 99.84% |
| Purchase = 0 (No Buy) | 13 | **0.16%** |

An unusually high conversion rate like this reframes the whole problem: the interesting, business-relevant prediction target isn't "who will buy" (almost everyone does) — it's identifying the small number of anomalous sessions that **don't** convert. With only 13 real positive examples in the entire dataset, every methodological choice below (splitting, resampling, evaluation) exists specifically to produce a trustworthy answer despite that scarcity.

## 3. Objectives

1. Build a model capable of reliably identifying the rare non-converting sessions among an overwhelmingly high-converting user base.
2. Compare Logistic Regression, Decision Tree, and Random Forest under a methodology that does not reward simply predicting the majority class.
3. Diagnose and correct the standard pitfalls of single-split evaluation and accuracy/ROC-AUC reporting under 0.16% prevalence.
4. Identify which behavioral features actually differentiate non-buying sessions, and turn that into an actionable business recommendation.

## 4. Tools and Technologies

- **Language & core libraries:** Python, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** Scikit-learn — `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
- **Preprocessing:** Scikit-learn — `StandardScaler`, `LabelEncoder`, `PCA`, `train_test_split`, `RepeatedStratifiedKFold`
- **Imbalance handling:** imbalanced-learn — `SMOTE`
- **Evaluation:** Scikit-learn `metrics` — `accuracy_score`, `f1_score`, `roc_auc_score`, `average_precision_score` (PR-AUC), `matthews_corrcoef` (MCC), `recall_score`, `precision_score`, `confusion_matrix`, `classification_report`

## 5. Preprocessing Pipeline

1. **Deduplication:** exact duplicate rows dropped.
2. **Identifier removal:** `user_id` dropped (not predictive).
3. **Missing values:** numeric columns imputed with the median, categorical columns imputed with the mode.
4. **Outlier check:** IQR-based bound check run on all continuous features (reported, not removed — a deliberate choice, since aggressively trimming outliers on a dataset with only 13 positive examples risks discarding real minority-class signal).
5. **Encoding:** `gender` label-encoded; `device_type` one-hot encoded (`device_Desktop`, `device_Mobile`, `device_Tablet`); boolean columns cast to int.
6. **Feature scaling:** `StandardScaler` applied to the continuous features (`age`, `time_on_site`, `pages_viewed`, `previous_purchases`, `cart_items`, `avg_session_time`, `bounce_rate`).
7. **SMOTE (Synthetic Minority Over-sampling Technique):** applied to the **training data only**, with `sampling_strategy=0.4` — a deliberately **partial** rebalancing (minority class raised to 40% of the majority class, not a full 1:1 balance). With only ~10–11 real non-buyer examples available in any given training split, generating enough synthetic points to reach full parity would mean the model trains on a class that is almost entirely synthetic interpolation rather than real behavior — a real overfitting risk. Partial balancing is the more conservative, defensible choice. In the cross-validated pipeline (see Section 6), SMOTE's `k_neighbors` is additionally **recalculated per fold**, since the number of real minority examples landing in training varies slightly fold to fold.

## 6. Modeling and Evaluation Approach

### 6.1 Phase 1 — Single 80/20 Stratified Split (initial approach)

The original evaluation used one stratified 80/20 train/test split. The stratified test set (1,600 sessions) contained only **3** real non-buyers — and this is precisely where the approach broke down.

| Model | Accuracy | ROC-AUC | Weighted F1 | No-Buy Recall | No-Buy Precision |
|---|---|---|---|---|---|
| Logistic Regression | 99.69% | 99.85% | 99.74% | 0.67 (2/3) | 0.33 |
| Decision Tree | 99.81% | 83.23% | 99.79% | 0.33 (1/3) | 0.50 |
| **Random Forest** | **99.81%** | **99.79%** | 99.72% | **0.00 (0/3)** | **0.00** |

**Why this failed:** Random Forest scored 99.81% accuracy and 99.79% ROC-AUC — both near-perfect — while catching **zero** of the three real non-buying sessions. With 99.84% of all sessions being purchases, a model can ignore the minority class completely and still clear 99% accuracy. Worse, with only 3 minority examples in the test set, recall could only ever land on 0/3, 1/3, 2/3, or 3/3 — a single misclassified session shifts the reported score by 33 points. This is not a stable measurement; it's the outcome of 3 coin flips, and a different random split could have told a completely different story.

### 6.2 Phase 2 — Repeated Stratified 5-Fold Cross-Validation (corrected approach)

To fix this, the single split was replaced with a **`RepeatedStratifiedKFold`** (5 folds × 20 repeats = **100 total evaluations**):

- **Stratified**, so every fold preserves the true 0.16%/99.84% ratio.
- **5 folds specifically**, because with only 13 total minority examples, 5 folds keeps a workable ~2–3 non-buyers per test fold and ~10–11 in training; a higher fold count (e.g. 10) would leave only ~1 per test fold, making per-class metrics degenerate far more often.
- **20 repeats (100 folds total)**, because a single 5-fold pass is still one arbitrary way of grouping 13 rare cases — repeating with different shuffles and averaging produces a stable mean **and** a standard deviation, i.e. an actual confidence range rather than one potentially-lucky number.
- Per fold: SMOTE (`k_neighbors` adapted to that fold's available minority count) is fit **only** on the training portion, and a fresh `StandardScaler` is likewise fit only on training data and applied to the untouched test fold — preventing any leakage from test to train.

## 7. Visualizations

**Logistic Regression — Feature Coefficients**
Cart items, average session time, and previous purchases carry the strongest positive coefficients (i.e., most associated with completing a purchase); bounce rate is the strongest negative coefficient (most associated with not purchasing).

![Logistic Regression Feature Coefficients](images/lr_coefficients.png)

**Random Forest — Feature Importance (200 Trees)**
Cart items, bounce rate, and previous purchases dominate; device type and ad-click features contribute almost nothing.

![Random Forest Feature Importance](images/rf_feature_importance.png)

**Decision Tree — Feature Importance**
An even more concentrated picture: cart items alone accounts for roughly two-thirds of total importance, with previous purchases and bounce rate making up almost all of the rest.

![Decision Tree Feature Importance](images/dt_feature_importance.png)

**Decision Tree — Purchase Decision Rules (Top 3 Levels)**
The learned tree splits first on `cart_items`, then on `previous_purchases`, `bounce_rate`, and `avg_session_time` — sessions with very low cart activity and low prior purchases are the ones the tree associates with "No Buy."

![Decision Tree Structure](images/decision_tree.png)

**3D PCA — Customer Behavior (exploratory only)**
A 3-component PCA was run on the SMOTE-resampled training data (57.44% combined variance explained: PC1 33.0%, PC2 12.7%, PC3 11.7%). The leading loadings on PC1 are `time_on_site`, `avg_session_time`, `cart_items`, and `bounce_rate`.

![3D PCA Customer Behavior](images/pca_3d.png)

> **Methodological note on this chart:** because it is fit on the SMOTE-resampled training set, the visually tight, thread-like blue cluster is largely composed of **synthetic** minority points (SMOTE generates new points by interpolating along lines between real minority neighbors, which produces exactly this stranded, radiating shape) rather than 2,556 independent real customers — only ~10 of those points are real non-buyers. For comparison, PCA run on the raw, untouched data explains only 45.3% of variance (vs. 57.44% on the resampled data), confirming that resampling meaningfully changes the captured variance structure. PCA is also unsupervised — its loadings reflect which features vary the most overall, not which features best separate buyers from non-buyers. This chart is retained here as **exploratory context**, not as confirmatory evidence of the drivers below; that role belongs to the supervised feature-importance/coefficient results above.

## 8. Metrics Results

### 8.1 Cross-Validated Summary (100 folds, mean ± standard deviation)

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|
| ROC-AUC | 0.999 ± 0.001 | 0.725 ± 0.140 | 0.988 ± 0.033 |
| **PR-AUC (No Buy)** | **0.739 ± 0.194** | 0.093 ± 0.073 | 0.485 ± 0.244 |
| **MCC** | **0.538 ± 0.130** | 0.212 ± 0.146 | 0.327 ± 0.308 |
| Weighted F1 | 0.997 ± 0.001 | 0.997 ± 0.001 | 0.998 ± 0.001 |
| Accuracy | 0.997 ± 0.001 | 0.996 ± 0.001 | 0.999 ± 0.001 |
| **Recall (No Buy)** | **0.897 ± 0.194** | 0.332 ± 0.235 | 0.228 ± 0.227 |
| Precision (No Buy) | 0.336 ± 0.132 | 0.146 ± 0.114 | 0.487 ± 0.466 |

All 100 folds completed successfully (0 skipped).

### 8.2 Why PR-AUC and MCC, specifically

- **PR-AUC (Average Precision) on "No Buy":** computed with the rare class relabeled as positive (`P(No Buy) = 1 − P(Buy)`); unlike ROC-AUC, it is directly sensitive to false alarms relative to the very small number of true positives, and its random-chance baseline is simply the class prevalence (0.16%) — giving a clean reference point. Logistic Regression's 0.739 is **~455× above that 0.16% baseline**.
- **MCC (Matthews Correlation Coefficient):** a single −1-to-+1 score incorporating all four confusion-matrix outcomes at once; it stays meaningful at extreme prevalence, unlike accuracy or F1, both of which remain ≥99.6% for every model here regardless of real minority-class performance.
- **Per-class recall/precision:** translate model behavior into business terms — recall answers "of all real non-buyers, how many did we catch?"; precision answers "of everything we flagged, how much was correct?"

## 9. Key Findings

- **Logistic Regression is the only model that reliably detects non-buying sessions** under rigorous evaluation: 89.7% average recall, PR-AUC 0.739 (~455× above chance), MCC 0.538.
- **Random Forest's single-split result was not a fluke of that one split.** Cross-validation confirms it structurally underperforms on the minority class (22.8% average recall) despite matching or exceeding Logistic Regression on accuracy/ROC-AUC — the same failure mode, now demonstrated at scale across 100 folds.
- **Accuracy and weighted F1 remain uninformative** (≥99.6% for all three models) regardless of evaluation method — only the imbalance-aware metrics (PR-AUC, MCC, per-class recall/precision) differentiate the models meaningfully.
- **Cart activity, prior purchase history, and session engagement (avg. session time, bounce rate)** are the consistent top predictors of non-purchase across Logistic Regression coefficients, Decision Tree rules, and Random Forest importance — device type and ad-click features contribute almost nothing.
- **Fold-to-fold variance is substantial** for every model (e.g., Logistic Regression's PR-AUC ranges roughly ±0.194) — an expected, disclosed consequence of only 13 real minority examples existing in the entire dataset, not a flaw in the methodology.

## 10. Recommendations

1. **Deploy Logistic Regression**, not Random Forest or Decision Tree, as the model for flagging at-risk (non-converting) sessions — it is the only one with both statistically validated minority recall and a meaningful MCC.
2. **Treat its output as a triage signal, not an automated trigger:** at 33.6% average precision on the minority class, roughly two of every three flagged sessions will be false alarms, so any downstream action taken on a flag (manual review, retention outreach, UX/fraud investigation) should be low-cost per instance.
3. **Prioritize on-site engagement levers** (cart activity, session engagement, bounce rate) **over acquisition-side levers** (ad targeting, demographic segmentation) when designing interventions — these are the variables that actually differentiate outcomes in this data.
4. **Keep collecting labeled non-buyer examples.** With only 13 in the current dataset, every metric in this report — however carefully validated — remains bounded by that small absolute sample size; more data would tighten every confidence interval reported here.

## 11. Conclusion

The project set out to build a defensible model for identifying rare non-converting e-commerce sessions without being misled by the inflated accuracy/ROC-AUC that 0.16% prevalence produces almost by default. That objective was reached: moving from a single 80/20 split — which reported near-perfect scores for a Random Forest model that caught zero real non-buyers — to a repeated stratified 5-fold cross-validation (100 folds) evaluated with imbalance-appropriate metrics revealed Logistic Regression as a genuinely reliable model, catching roughly 9 in 10 real non-buying sessions with a PR-AUC 455 times above random chance. The one caveat that should travel with this conclusion: the entire analysis rests on 13 real minority examples, and no evaluation methodology, however rigorous, fully substitutes for more labeled data — the standard deviations reported throughout this document exist specifically to make that uncertainty visible rather than hide it.

## 12. Limitations and Notes for Future Work

- Only 13 real non-buyer examples exist in the source data; all reported metrics carry correspondingly wide confidence ranges.
- The PCA visualization (Section 7) is exploratory: it is fit on SMOTE-resampled data and should not be read as independent confirmation of feature importance.
- Feature-importance/coefficient analysis (Section 7) was performed on a single train/test split; a natural next step is to aggregate feature importance across all 100 cross-validation folds for full consistency with the performance-metric methodology.
- Future work: collect additional real non-buyer sessions over time; re-run this evaluation as the sample size grows to narrow the reported confidence intervals.
