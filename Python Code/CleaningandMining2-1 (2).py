# _______________ Step 0: Import Libraries ──────────────────────────
import pandas as pd
import numpy as np

# Visualization (for outliers)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
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
    roc_auc_score
)

# _______________ Clustering ───────────────────────────────────────
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ____________________ Load dataset ____________________
df = pd.read_csv(r"C:\Users\ramez\Downloads\ecommerce_user_behavior_8000.csv")

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
df.to_csv("ecommerce_user_behavior_cleaned.csv", index=False)
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
# ── Per-model evaluation ─────────────────────────────────────

# Dictionary of prediction arrays (for metrics)
models = {
    'Logistic Regression': y_pred_lr,
    'Decision Tree':       y_pred_dt,
    'Random Forest':       y_pred_rf,
}
 
# Dictionary of fitted estimators (needed for predict_proba / ROC-AUC)
fitted = {
    'Logistic Regression': lr,
    'Decision Tree':       dt,
    'Random Forest':       rf,
}
 
for name, pred in models.items():
    print(f'\n=== {name} ===')
    print(f'Accuracy : {accuracy_score(y_test, pred):.4f}')
    print(f'F1 Score : {f1_score(y_test, pred, average="weighted"):.4f}')
    y_prob = fitted[name].predict_proba(X_test)[:, 1]
    print(f'ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}')
    print(classification_report(y_test, pred,
                        target_names=['No Buy', 'Buy']))
 
# ── Confusion matrices — all 3 models side by side ───────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
 
for ax, (name, pred) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False,
               xticklabels=['No Buy','Buy'],
               yticklabels=['No Buy','Buy'])
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Predicted')  
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices — All Models', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
 


 
