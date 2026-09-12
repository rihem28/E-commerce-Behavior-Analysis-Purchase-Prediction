# _______________ Import Libraries ──────────────────────────
import pandas as pd
import numpy as np

# Visualization (for outliers)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Handling class imbalance
from imblearn.over_sampling import SMOTE

# _______________ Models ────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# _______________ Evaluation ───────────────────────────────────────
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    recall_score,
    precision_score
)

# _______________ Clustering ───────────────────────────────────────
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ____________________ Load dataset ____________________
df = pd.read_csv(r"C:\Users\abdel\E-commerce-Behavior-Analysis-Purchase-Prediction\Data\Raw Data\ecommerce_user_behavior_8000.csv")

# Display dataset
print(f"Shape: {df.shape}")
print(df.head())
# Check Missing Values
print(df.isnull().sum())

# ____________________ Remove duplicates ____________________
df = df.drop_duplicates().reset_index(drop=True)

# ____________________ Drop identifier ____________________
df = df.drop(columns=['user_id'])

# ____________________ Handle Missing Values ____________________

# Fill numeric values with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical values with mode
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# _________Continuous features to check for outliers__________
num_cols = ['age', 'time_on_site', 'pages_viewed',
             'previous_purchases', 'cart_items',
             'avg_session_time', 'bounce_rate']
 
for col in num_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f'{col}: {n_out} outliers')

# ____________________ Encoding ____________________
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

df = pd.get_dummies(df, columns=['device_type'], prefix='device')

#_______________ Convert bool to int_________________________
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# ____________Save cleaned dataset___________
df.to_csv("cleaned_ecommerce_user_behavior.csv", index=False)
print("Cleaned dataset saved successfully!")

# ____________________ Separate features and target ____________________
X = df.drop(columns=['purchase'])
y = df['purchase'].astype(int)

print("Before balancing:")
print(y.value_counts())

# ____________________ Train Test Split ____________________
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ____________________ SMOTE (Partial Balancing) ____________________
smote = SMOTE(
    sampling_strategy=0.4,   # 40% balance (not equal)
    random_state=42
)

X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train_bal.value_counts())

# ____________________ Feature Scaling ____________________
scale_cols = [
    'age', 'time_on_site', 'pages_viewed',
    'previous_purchases', 'cart_items',
    'avg_session_time', 'bounce_rate'
]

scaler = StandardScaler()

X_train_bal[scale_cols] = scaler.fit_transform(X_train_bal[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# ================== MINING STARTS HERE ==================

# ── Logistic Regression ───────────────────────────────────────────
lr = LogisticRegression(class_weight=None, max_iter=1000, random_state=42)
lr.fit(X_train_bal, y_train_bal)
y_pred_lr = lr.predict(X_test)
 
# Feature coefficients — direction and strength of each variable
coef_df = pd.DataFrame({'Feature': X_train_bal.columns,'Coefficient': lr.coef_[0]})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print(coef_df)
 
# Visualise coefficients
plt.figure(figsize=(9, 5))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.title('Logistic Regression — Feature Coefficients')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig('lr_coefficients.png', dpi=150)
plt.show()
 
# ── Decision Tree ─────────────────────────────────────────────────
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt.fit(X_train_bal, y_train_bal)
y_pred_dt = dt.predict(X_test)
 
# Visualise the tree structure (top 3 levels for readability)
plt.figure(figsize=(20, 8))
plot_tree(dt, max_depth=3, feature_names=X_train_bal.columns,
             class_names=['No Buy', 'Buy'], filled=True, rounded=True)
plt.title('Decision Tree — Purchase Decision Rules (Top 3 Levels)', fontsize=13)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()
 
# Feature importance bar chart
fi_dt = pd.DataFrame({'Feature': X_train_bal.columns,'Importance': dt.feature_importances_})
fi_dt = fi_dt.sort_values('Importance', ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(data=fi_dt, x='Importance', y='Feature', palette='Blues_r')
plt.title('Decision Tree — Feature Importance')
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=150)
plt.show()

# ── Random Forest ───────────────────────────────────────────────── 
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)
y_pred_rf = rf.predict(X_test)
 
# Feature importance — averaged across all 200 trees
fi_rf = pd.DataFrame({'Feature': X_train_bal.columns,'Importance': rf.feature_importances_})
fi_rf = fi_rf.sort_values('Importance', ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(data=fi_rf, x='Importance', y='Feature', palette='Blues_r')
plt.title('Random Forest — Feature Importance (200 Trees)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=150)
plt.show()


# ── 3D PCA (Balanced Data) ──────────────────────────
from mpl_toolkits.mplot3d import Axes3D

pca3 = PCA(n_components=3, random_state=42)
X_pca3 = pca3.fit_transform(X_train_bal)

explained_var = pca3.explained_variance_ratio_
print(f"PC1: {explained_var[0]:.4f}")
print(f"PC2: {explained_var[1]:.4f}")
print(f"PC3: {explained_var[2]:.4f}")
print(f"Total Explained: {explained_var.sum():.4f}")

# DataFrame
pca3_df = pd.DataFrame({
    'PC1': X_pca3[:, 0],
    'PC2': X_pca3[:, 1],
    'PC3': X_pca3[:, 2],
    'Purchase': y_train_bal.values
})

# Sample to avoid clutter
pca3_sample = pca3_df.sample(3000, random_state=42)

# Plot 3D
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    pca3_sample['PC1'],
    pca3_sample['PC2'],
    pca3_sample['PC3'],
    c=pca3_sample['Purchase'],
    cmap='coolwarm',
    alpha=0.6
)

ax.set_title('3D PCA — Customer Behavior')
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)')
ax.set_zlabel(f'PC3 ({explained_var[2]*100:.1f}%)')

plt.tight_layout()
plt.savefig('pca_3d.png', dpi=150)
plt.show()
# ── PCA Feature Importance (Loadings) ──────────────────────────

loadings = pd.DataFrame(
    pca3.components_.T,
    columns=['PC1', 'PC2', 'PC3'],
    index=X_train_bal.columns
)

# Focus on PC1 (most important)
pc1_sorted = loadings['PC1'].abs().sort_values(ascending=False)

print("\nTop features influencing behavior (PC1):")
print(pc1_sorted.head(10))

# ================== CROSS-VALIDATED EVALUATION (IMBALANCE-AWARE) ==================
# Replaces the single 80/20 split evaluation, which had only 3 non-buyers in the
# test set — too few for a statistically reliable estimate of minority-class performance.

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

cv_models = {
    'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree':        lambda: DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
    'Random Forest':        lambda: RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
}

cv_results = {name: {'roc_auc': [], 'pr_auc': [], 'mcc': [], 'f1_weighted': [],
                      'accuracy': [], 'recall_minority': [], 'precision_minority': []}
              for name in cv_models}

for train_idx, test_idx in cv.split(X, y):
    X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_tr, y_te = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    # Adapt SMOTE's k_neighbors to however many minority samples this fold's
    # training set actually has (some folds may have very few).
    n_min = y_tr.value_counts().min()
    if n_min >= 2:
        k = min(5, n_min - 1)
        sm = SMOTE(sampling_strategy=0.4, random_state=42, k_neighbors=k)
        X_tr_bal, y_tr_bal = sm.fit_resample(X_tr, y_tr)
    else:
        X_tr_bal, y_tr_bal = X_tr, y_tr

    fold_scaler = StandardScaler()
    X_tr_bal[scale_cols] = fold_scaler.fit_transform(X_tr_bal[scale_cols])
    X_te[scale_cols] = fold_scaler.transform(X_te[scale_cols])

    if len(set(y_te)) < 2:
        continue  # skip folds where the test split has only one class present

    for name, make_model in cv_models.items():
        model = make_model()
        model.fit(X_tr_bal, y_tr_bal)
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1]

        cv_results[name]['roc_auc'].append(roc_auc_score(y_te, prob))
        # PR-AUC scored against the MINORITY class ("No Buy"), not the trivial majority class
        cv_results[name]['pr_auc'].append(average_precision_score((y_te == 0).astype(int), 1 - prob))
        cv_results[name]['mcc'].append(matthews_corrcoef(y_te, pred))
        cv_results[name]['f1_weighted'].append(f1_score(y_te, pred, average='weighted'))
        cv_results[name]['accuracy'].append(accuracy_score(y_te, pred))
        cv_results[name]['recall_minority'].append(recall_score(y_te, pred, pos_label=0, zero_division=0))
        cv_results[name]['precision_minority'].append(precision_score(y_te, pred, pos_label=0, zero_division=0))

# ── Build summary table (mean ± std across all folds) ──────────────────────
summary_rows = []
for name, metrics in cv_results.items():
    row = {'Model': name}
    for metric, vals in metrics.items():
        vals = np.array(vals)
        row[metric] = f"{vals.mean():.3f} ± {vals.std():.3f}"
    summary_rows.append(row)

cv_summary_df = pd.DataFrame(summary_rows).set_index('Model')
print("\n=== Cross-Validated Results (100 folds) ===")
print(cv_summary_df)

cv_summary_df.to_csv("cv_results_summary.csv")
print("\nSaved cross-validated summary to cv_results_summary.csv")
 


 
