"""
E-commerce Pipeline
====================================================================
Predicts whether a web session ends in a purchase from 8,000 simulated sessions
in which only 13 are non-buyers (the minority class, purchase = 0).

Design rules
  * every number AND every graph comes from cross-validation 
  (or from models fitted on all data after the comparison).
  * Imputation, scaling, encoding and SMOTE all live inside one pipeline, so each
    is learned from the training part of a fold only (no leakage).
  * Rows with an unknown target are removed, never imputed.

Outputs:  figures/  (PNG charts)   results/  (CSV tables + final model)
          cv_cache/ (cached cross-validation scores; delete it after changing code)
"""
# =====================================================================
# 1. Imports and Libraries
# =====================================================================
import os
import textwrap
from pathlib import Path
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score, 
    make_scorer, 
    matthews_corrcoef,
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    )

from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_predict, cross_validate)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# =====================================================================
# 2. CONFIGURATION
# =====================================================================
DATA_PATH = Path(os.environ.get(
    "ECOM_DATA",
    r"C:\Users\abdel\E-commerce-Behavior-Analysis-Purchase-Prediction\Data\ecommerce_user_behavior_8000.csv"
    ))

RANDOM_STATE = 42

# configure validation
N_SPLITS = 5                                                # folds per repeat
N_REPEATS = int(os.environ.get("ECOM_REPEATS", 20))         # 5 x 20 = 100 train/test evaluations
OOF_REPEATS = int(os.environ.get("ECOM_OOF_REPEATS", 5))    # repeats for out-of-fold curves

# configure SMOTE
SMOTE_RATIO = 0.4                                           # minority = 40% of majority after resampling
K_NEIGHBORS = 5                                             # SMOTE neighbours (needs >= 6 minority rows per training fold)

RUN_ABLATIONS = True                                        # if False = main configuration only (much faster)
USE_CACHE = True                                            # reuse saved cross-validation scores; delete cv_cache/ if code changes
SHOW_PLOTS = True                                           # True = save and open each figure on screen

FIG_DIR, OUT_DIR, CACHE_DIR = Path("figures"), Path("results"), Path("cv_cache")

TARGET = "purchase"

CONTINUOUS = ["age", "time_on_site", "pages_viewed", 
              "previous_purchases", "cart_items", 
              "avg_session_time", "bounce_rate"]

FLAGS = ["discount_seen", "ad_clicked", "returning_user"]

CATEGORICAL = ["gender", "device_type"]
GENDER_LEVELS = ["Female", "Male"]
DEVICE_LEVELS = ["Desktop", "Mobile", "Tablet"]

# Column order produced by the preprocessor: continuous, flags, one-hot columns
FEATURE_NAMES = CONTINUOUS + FLAGS + ["gender_Male", 
"device_type_Mobile", "device_type_Tablet"]

CAT_IDX = list(range(len(CONTINUOUS), len(FEATURE_NAMES)))            # binary columns, used by SMOTENC

OUTCOME_ORDER = ["Buyer", "Non-buyer"]
OUTCOME_COLORS = {"Buyer": "#5B8DB8", "Non-buyer": "#D6403E"}
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]
MODEL_COLORS = {"Logistic Regression": "#1B9E8F", 
                "Decision Tree": "#E8A33D", "Random Forest": "#5B5FC7"}

VARIANTS = {                                                             # tag -> (label, use class_weight, resampler)
    "main":        ("SMOTE + class weight", True, "smote"),
    "smote_only":  ("SMOTE only", False, "smote"),
    "smotenc":     ("SMOTENC + class weight", True, "smotenc"),
    "weight_only": ("Class weight only (no resampling)", True, None),
}

# =====================================================================
# 3. PLOT STYLE AND SAVE HELPER
# =====================================================================
def apply_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "#444444",
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 10,
        "axes.labelsize": 11, "axes.labelcolor": "#222222",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E3E6EA", "grid.linewidth": 0.8, "legend.frameon": False,
        "xtick.color": "#333333", "ytick.color": "#333333", "font.family": "DejaVu Sans"})

def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  saved {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

# =====================================================================
# 4. PIPELINE, MODELS AND SCORERS
# =====================================================================
def make_preprocessor():
    """Impute + scale + encode. 
    Fitted on the training part of each fold only."""
    continuous = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    flags = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[GENDER_LEVELS, DEVICE_LEVELS], 
            drop="first",
            handle_unknown="ignore", 
            sparse_output=False
        ))
    ])
    return ColumnTransformer(
        [("continuous", continuous, CONTINUOUS),
         ("flags", flags, FLAGS),
         ("categorical", categorical, CATEGORICAL)],
        verbose_feature_names_out=False)

def make_sampler(kind):
    if kind == "smote":
        return SMOTE(
            sampling_strategy=SMOTE_RATIO, 
            k_neighbors=K_NEIGHBORS, 
            random_state=RANDOM_STATE
        )
    if kind == "smotenc":
        return SMOTENC(
            categorical_features=CAT_IDX, 
            sampling_strategy=SMOTE_RATIO,
            k_neighbors=K_NEIGHBORS, 
            random_state=RANDOM_STATE
        )
    return None

def make_pipe(clf, sampler="smote"):
    """prep -> (optional) resampler -> classifier. 
    The resampler runs only during fit."""
    steps = [("prep", make_preprocessor())]
    resampler = make_sampler(sampler)
    if resampler is not None:
        steps.append(("resample", resampler))   
    steps.append(("clf", clf))
    return ImbPipeline(steps)

def make_models(class_weight=True):
    cw = "balanced" if class_weight else None
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            class_weight=cw, 
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, 
            class_weight=cw, 
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            class_weight=cw,
            random_state=RANDOM_STATE,
            n_jobs=1
        )}

def pr_auc_minority(estimator, X_te, y_te):
    """PR-AUC for the minority class (No Buy = 0),
    read from that class's probability."""
    col = list(estimator.classes_).index(0)
    return average_precision_score(
        (y_te == 0).astype(int), 
        estimator.predict_proba(X_te)[:, col]
    )

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": pr_auc_minority,
    "mcc": make_scorer(matthews_corrcoef),
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "recall_minority": make_scorer(recall_score, 
    pos_label=0, zero_division=0),
    "precision_minority": make_scorer(precision_score, 
    pos_label=0, zero_division=0),
}

# =====================================================================
# 3. CROSS-VALIDATION HELPERS (with a small on-disk cache)
# =====================================================================
def cached(name, compute):
    path = CACHE_DIR / f"{name}.pkl"
    if USE_CACHE and path.exists():
        print(f"  [cache] loaded {name}")
        return pd.read_pickle(path)
    out = compute()
    if USE_CACHE:
        CACHE_DIR.mkdir(exist_ok=True)
        pd.to_pickle(out, path)
    return out

def run_cv(X, y, cv, tag, class_weight, sampler, fingerprint):
    """Cross-validate the three models; 
    returns {model: {metric: array of fold scores}}."""
    results = {}
    for name, clf in make_models(class_weight).items():
        def compute(clf=clf):
            res = cross_validate(
                make_pipe(clf, sampler), 
                X, 
                y, 
                cv=cv, 
                scoring=SCORING,
                n_jobs=-1, 
                error_score="raise"
            )
            return {m: np.asarray(res[f"test_{m}"]) for m in SCORING}
        key = f"cv_{tag}_{name.replace(' ', '_')}_{N_REPEATS}x{N_SPLITS}_{fingerprint}"
        results[name] = cached(key, compute)
    return results

def summarize(results):                    # "mean ± std" text table
    return pd.DataFrame(
        {n: {m: f"{r[m].mean():.3f} ± {r[m].std():.3f}" for m in SCORING}
         for n, r in results.items()}).T

def means(results):                        # numeric means
    return pd.DataFrame(
        {n: {m: r[m].mean() for m in SCORING} 
         for n, r in results.items()}).T

def paired_comparison(results, reference="Logistic Regression",
                      metrics=("pr_auc", "recall_minority", "mcc", "precision_minority")):
    rows = []
    for other in results:
        if other == reference:
            continue
        for m in metrics:
            diff = results[reference][m] - results[other][m]
            rows.append({
                "metric": m, 
                "vs": other, 
                "mean_difference": diff.mean(),
                "reference_better_pct": (diff > 0).mean() * 100,
                "tied_pct": (diff == 0).mean() * 100
            })     
    return pd.DataFrame(rows)

def oof_scores(X, y, fingerprint):
    """Out-of-fold P(No Buy) for each model, 
    OOF_REPEATS times (every row scored once per repeat)."""
    out = {}
    for name, clf in make_models(True).items():
        def compute(clf=clf):
            rows = []
            for r in range(OOF_REPEATS):
                skf = StratifiedKFold(
                    n_splits=N_SPLITS, 
                    shuffle=True, 
                    random_state=RANDOM_STATE + r
                )
                proba = cross_val_predict(
                    make_pipe(clf, "smote"), 
                    X, 
                    y, 
                    cv=skf,
                    method="predict_proba", 
                    n_jobs=-1
                    )
                rows.append(proba[:, 0])            # column 0 = class 0 = No Buy
            return np.vstack(rows)
        
        out[name] = cached(f"oof_{name.replace(' ', '_')}_{OOF_REPEATS}_{fingerprint}", compute)
    return out

def fold_models(X, y, clf, sampler, cv_obj):
    """Fit a pipeline on every training fold 
    and return the fitted classifiers."""
    res = cross_validate(
        make_pipe(clf, sampler), 
        X, 
        y, 
        cv=cv_obj, 
        scoring="roc_auc",
        return_estimator=True, 
        n_jobs=1
    )
    return [e.named_steps["clf"] for e in res["estimator"]]

def threshold_table(scores, y_min, thresholds):
    rows = []
    for t in thresholds:
        flagged = scores >= t                                   # (repeats, n_rows)
        tp = (flagged & (y_min == 1)).sum(axis=1)
        n_flag = flagged.sum(axis=1)

        rows.append({"threshold": t,
                     "recall": (tp / y_min.sum()).mean(),
                     "precision": np.where(n_flag > 0, tp / np.maximum(n_flag, 1), np.nan).mean(),
                     "flagged_per_run": n_flag.mean(),
                     "caught_per_run": tp.mean()
                    })
        
    return pd.DataFrame(rows)

# =====================================================================
# 4. FIGURES
# =====================================================================
def fig_missing(raw):
    miss = raw.isna().sum().sort_values()
    pct = miss / len(raw) * 100
    colors = [OUTCOME_COLORS["Non-buyer"] if c == TARGET else "#9AA7B4" for c in miss.index]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(miss.index, pct.values, color=colors, height=0.68)
    for i, (col, n) in enumerate(miss.items()):
        ax.text(pct[col] + 0.03, i, f"{n} ({pct[col]:.1f}%)", va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, max(pct.max() * 1.3, 1))
    ax.set_xlabel("Rows with a missing value (%)")
    ax.set_title("Missing values per column (raw file)")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.02, "Red = the target. Rows without a label are removed, not imputed.",
            transform=ax.transAxes, ha="right", fontsize=9, color="#555555")
    save_fig(fig, "01_missing_values")

def fig_class_balance(y):
    counts = y.value_counts().reindex([1, 0])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(OUTCOME_ORDER, counts.values, width=0.55,
                  color=[OUTCOME_COLORS[k] for k in OUTCOME_ORDER])
    ax.set_yscale("log")

    ax.set_ylim(5, counts.max() * 6)
    for b, n in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, n * 1.25, f"{n:,}\n({n / len(y) * 100:.2f}%)",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Sessions (log scale)")
    ax.set_title("Extreme class imbalance")
    ax.set_xlabel(f"Always predicting 'Buyer' gives {counts[1] / len(y) * 100:.2f}% accuracy "
                  "but finds 0% of non-buyers", fontsize=10, color="#555555")
    save_fig(fig, "02_class_imbalance")

def fig_profiles(X, y):
    data = X[CONTINUOUS].copy()
    data["Outcome"] = np.where(y.values == 1, "Buyer", "Non-buyer")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6))
    axes = axes.ravel()
    for ax, col in zip(axes, CONTINUOUS):
        sns.boxplot(data=data, x="Outcome", y=col, hue="Outcome", order=OUTCOME_ORDER,
                    hue_order=OUTCOME_ORDER, palette=OUTCOME_COLORS, width=0.55, fliersize=0,
                    saturation=0.65, linewidth=1.1, ax=ax, legend=False)
        sns.stripplot(data=data[data["Outcome"] == "Non-buyer"], x="Outcome", y=col,
                      order=OUTCOME_ORDER, color="#5E1514", size=5, jitter=0.12, alpha=0.9, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[-1].axis("off")
    axes[-1].text(0.0, 0.6, "Boxes: spread of each group.\nDark dots: the 13 individual\nnon-buyers.",
                  fontsize=11, color="#444444", va="center")
    fig.suptitle("Behaviour profile: buyers vs. non-buyers", fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, "03_feature_profiles")

def fig_cv_comparison(results):
    panels = [("pr_auc", "PR-AUC (No Buy)"), ("mcc", "MCC"),
              ("recall_minority", "Recall (No Buy)"), ("precision_minority", "Precision (No Buy)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (metric, label) in zip(axes, panels):
        long = pd.DataFrame([{"Model": n, "value": v} for n in MODEL_NAMES for v in results[n][metric]])
        sns.boxplot(data=long, x="Model", y="value", hue="Model", order=MODEL_NAMES,
                    hue_order=MODEL_NAMES, palette=MODEL_COLORS, width=0.55, fliersize=0,
                    saturation=0.7, ax=ax, legend=False)
        sns.stripplot(data=long, x="Model", y="value", order=MODEL_NAMES, color="black",
                      size=2.3, alpha=0.25, jitter=0.2, ax=ax)
        ax.scatter(range(3), [results[n][metric].mean() for n in MODEL_NAMES], marker="D", s=55,
                   color="white", edgecolor="black", zorder=6)
        ax.set_xticks(range(3))
        ax.set_xticklabels([n.replace(" ", "\n") for n in MODEL_NAMES])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(label)
    fig.suptitle(f"Cross-validated performance across {len(results[MODEL_NAMES[0]]['mcc'])} folds "
                 "(diamond = mean, dots = individual folds)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "04_cv_model_comparison")

def fig_operating_points(oof, y):
    y_min = (y.values == 0).astype(int)
    n_rep = oof[MODEL_NAMES[0]].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    ax = axes[0]
    for name in MODEL_NAMES:
        scores, truth = oof[name].ravel(), np.tile(y_min, n_rep)
        prec, rec, _ = precision_recall_curve(truth, scores)
        ap = average_precision_score(truth, scores)
        ax.step(rec, prec, where="post", color=MODEL_COLORS[name], lw=2.3, label=f"{name} (AP = {ap:.2f})")
    ax.axhline(y_min.mean(), color="#888888", ls="--", lw=1.2, label=f"Random guess ({y_min.mean():.4f})")
    ax.set_xlabel("Recall (No Buy)")
    ax.set_ylabel("Precision (No Buy)")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, 1.0)
    ax.set_title("Precision-recall curves (out-of-fold)")
    ax.legend(loc="upper right", fontsize=9)

    lr_scores = oof["Logistic Regression"]
    table = threshold_table(lr_scores, y_min, np.round(np.arange(0.05, 0.96, 0.05), 2))
    ax = axes[1]
    ax.plot(table["threshold"], table["recall"], color="#1B9E8F", lw=2.3, marker="o", ms=4, label="Recall")
    ax.plot(table["threshold"], table["precision"], color="#D6403E", lw=2.3, marker="o", ms=4, label="Precision")
    ax.axvline(0.5, color="#555555", ls="--", lw=1.2)
    ax.text(0.505, 0.04, "default 0.5", fontsize=9, color="#555555")
    ax.set_xlabel("Decision threshold on P(No Buy)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.03)
    ax.set_title("Logistic Regression: threshold trade-off")
    ax.legend(loc="center left", fontsize=9)

    pred = lr_scores >= 0.5
    cm = np.array([[(pred & (y_min == 1)).sum(axis=1).mean(), (~pred & (y_min == 1)).sum(axis=1).mean()],
                   [(pred & (y_min == 0)).sum(axis=1).mean(), (~pred & (y_min == 0)).sum(axis=1).mean()]])
    share = cm / cm.sum(axis=1, keepdims=True)
    ax = axes[2]
    for i in range(2):
        for j in range(2):
            base = np.array(plt.matplotlib.colors.to_rgb("#1B9E8F" if i == j else "#D6403E"))
            strength = max(share[i, j], 0.12)
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.96, 0.96, color=1 - strength * (1 - base)))
            ax.text(j + 0.48, 1 - i + 0.55, f"{cm[i, j]:,.1f}", ha="center", va="center", fontsize=17,
                    fontweight="bold", color="white" if strength > 0.6 else "#222222")
            ax.text(j + 0.48, 1 - i + 0.25, f"{share[i, j] * 100:.1f}% of row", ha="center", va="center",
                    fontsize=9, color="white" if strength > 0.6 else "#444444")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.48, 1.48])
    ax.set_xticklabels(["Predicted\nNon-buyer", "Predicted\nBuyer"])
    ax.set_yticks([1.48, 0.48])
    ax.set_yticklabels(["Actual\nNon-buyer", "Actual\nBuyer"])
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Logistic Regression: average confusion matrix\n(sessions per full pass; teal = correct, red = error)")
    fig.tight_layout()
    save_fig(fig, "05_pr_threshold_confusion")
    return table

def fig_ablation(variant_means):
    panels = [("pr_auc", "PR-AUC (No Buy)"), ("mcc", "MCC"),
              ("recall_minority", "Recall (No Buy)"), ("precision_minority", "Precision (No Buy)")]
    rows = [{"Variant": textwrap.fill(label, 16), "Model": model, **{m: variant_means[tag].loc[model, m]
            for m, _ in panels}} for tag, (label, _, _) in VARIANTS.items() if tag in variant_means
            for model in MODEL_NAMES]
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for ax, (metric, label) in zip(axes.ravel(), panels):
        sns.barplot(data=data, x="Variant", y=metric, hue="Model", hue_order=MODEL_NAMES,
                    palette=MODEL_COLORS, ax=ax, edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(0, 1.0)
        ax.legend_.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sensitivity analysis: does the resampling / weighting choice matter?",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, "06_ablation")

def fig_lr_coefficients(coef_smote, coef_plain):
    order = np.argsort(coef_smote.mean(axis=0))
    names = np.array(FEATURE_NAMES)[order]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), sharey=True)
    panels = [(coef_smote, "SMOTE + class weight (the evaluated model)"),
              (coef_plain, "Class weight only (no synthetic rows)")]
    for ax, (coefs, title) in zip(axes, panels):
        m, s = coefs.mean(axis=0)[order], coefs.std(axis=0)[order]
        ax.barh(names, m, xerr=s, color=np.where(m > 0, "#1B9E8F", "#D6403E"), height=0.7,
                error_kw=dict(ecolor="#333333", lw=1.1, capsize=3))
        ax.axvline(0, color="black", lw=0.9)
        ax.set_title(title)
        ax.set_xlabel("Coefficient (log-odds of buying)")
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Logistic Regression coefficients: mean ± std over the {len(coef_smote)} training folds",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, "07_lr_coefficients")

def fig_tree(tree_clf):
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(
        tree_clf, max_depth=3, 
        feature_names=FEATURE_NAMES, 
        class_names=["No Buy", "Buy"],
        filled=True, rounded=True, 
        impurity=False, fontsize=10, 
        precision=2, ax=ax
    )
    ax.set_title("Decision Tree: top three levels (fitted on all data after SMOTE)", fontsize=15)
    fig.text(0.5, 0.06, "Thresholds are in standardised units (z-scores). 'value' = class-weighted sample "
             "counts, so it includes synthetic rows and balanced weights.", ha="center", fontsize=11, color="#555555")
    save_fig(fig, "08_decision_tree")

def fig_importances(imp_dt, imp_rf):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    panels = [(imp_dt, f"Decision Tree ({len(imp_dt)} folds)", MODEL_COLORS["Decision Tree"]),
              (imp_rf, f"Random Forest ({len(imp_rf)} folds)", MODEL_COLORS["Random Forest"])]
    for ax, (imp, title, color) in zip(axes, panels):
        order = np.argsort(imp.mean(axis=0))
        mean, std = imp.mean(axis=0)[order], imp.std(axis=0)[order]
        ax.barh(np.array(FEATURE_NAMES)[order], mean, xerr=[np.minimum(std, mean), std],            # importance >= 0
                color=color, height=0.7, error_kw=dict(ecolor="#333333", lw=1.1, capsize=3))
        ax.set_title(title)
        ax.set_xlabel("Importance (mean ± std across folds)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Tree-based feature importance", fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, "09_feature_importance")

def pca_view(Z_std, y):
    pca = PCA(random_state=RANDOM_STATE).fit(Z_std)
    scores, ev = pca.transform(Z_std), pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_[:3].T, 
        index=FEATURE_NAMES, 
        columns=["PC1", "PC2", "PC3"]
    )
    rng = np.random.default_rng(RANDOM_STATE)
    buyers = rng.choice(np.flatnonzero(y.values == 1), size=2500, replace=False)
    non_buyers = np.flatnonzero(y.values == 0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax = axes[0, 0]
    comp = np.arange(1, len(ev) + 1)
    ax.bar(comp, ev * 100, color="#8DA0B8", label="Per component")
    ax.plot(comp, np.cumsum(ev) * 100, color="#D6403E", marker="o", lw=2, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(comp)
    ax.set_title("Variance explained")
    ax.legend(loc="center right")

    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-0.7, vmax=0.7,
                linewidths=1, linecolor="white", cbar_kws={"shrink": 0.8}, ax=axes[0, 1])
    axes[0, 1].set_title("Feature loadings (first three components)")

    for ax, (i, j) in zip(axes[1], [(0, 1), (0, 2)]):
        ax.scatter(scores[buyers, i], scores[buyers, j], s=14, color=OUTCOME_COLORS["Buyer"], alpha=0.3,
                   label="Buyers (2,500 sampled)")
        ax.scatter(scores[non_buyers, i], scores[non_buyers, j], s=90, color=OUTCOME_COLORS["Non-buyer"],
                   edgecolor="black", lw=0.9, label="Non-buyers (all 13)", zorder=5)
        ax.set_xlabel(f"PC{i + 1} ({ev[i] * 100:.1f}%)")
        ax.set_ylabel(f"PC{j + 1} ({ev[j] * 100:.1f}%)")
        ax.set_title(f"PC{i + 1} vs PC{j + 1}")
        ax.legend(loc="center right", fontsize=9)
    fig.suptitle("PCA on the real (unresampled) sessions", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    save_fig(fig, "10_pca_overview")
    return scores, ev, loadings, buyers, non_buyers

def fig_pca3d(scores, ev, buyers, non_buyers):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(scores[buyers, 0], scores[buyers, 1], scores[buyers, 2], s=10,
               color=OUTCOME_COLORS["Buyer"], alpha=0.25, label="Buyers (sampled)")
    ax.scatter(scores[non_buyers, 0], scores[non_buyers, 1], scores[non_buyers, 2], s=85,
               color=OUTCOME_COLORS["Non-buyer"], edgecolor="black", lw=0.8, label="Non-buyers (all 13)")
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2] * 100:.1f}%)")
    ax.view_init(elev=20, azim=35)
    ax.set_title(f"3D PCA ({ev[:3].sum() * 100:.1f}% of total variance)")
    ax.legend(loc="upper left")
    save_fig(fig, "11_pca_3d")

# =====================================================================
# 5. MAIN WORKFLOW
# =====================================================================
def main():
    apply_style()
    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    # ---- 5.1 Load, inspect and clean ------------------------------------------------ 
    raw = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {raw.shape}")
    print(raw.isna().sum().to_string())
    fig_missing(raw)

    df = raw.drop(columns=["user_id"])                              # identifier, no behavioural signal
    n_start = len(df)
    df = df.drop_duplicates().reset_index(drop=True)                # after dropping the ID, so it can work
    n_dup = n_start - len(df)
    
    n_mid = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)          # unknown outcome: remove, never impute
    n_no_target = n_mid - len(df)
    print(f"Duplicate rows removed: {n_dup} | Rows removed for missing target: {n_no_target} "
          f"| Rows kept: {len(df)}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    assert set(CONTINUOUS + FLAGS + CATEGORICAL) == set(X.columns), "Unexpected columns in the data"
    assert list(make_preprocessor().fit(X).get_feature_names_out()) == FEATURE_NAMES
    print(f"Classes: Buyers = {(y == 1).sum()}, Non-buyers = {(y == 0).sum()} "
          f"({(y == 0).mean() * 100:.2f}%)")

    outliers = pd.Series({
        c: int(((X[c] < X[c].quantile(.25) - 1.5 * 
    (X[c].quantile(.75) - X[c].quantile(.25))) |
                (X[c] > X[c].quantile(.75) + 1.5 * 
    (X[c].quantile(.75) - X[c].quantile(.25)))).sum())
        for c in CONTINUOUS
    }, name="iqr_outliers")
    print("IQR outliers (reported only, rows kept):")
    print(outliers.to_string())

    pd.DataFrame({
        "item": ["raw rows", "duplicate rows removed", "rows removed (missing target)",
                 "rows used", "buyers", "non-buyers"],
        "value": [len(raw), n_dup, n_no_target, len(df), int((y == 1).sum()), 
                  int((y == 0).sum())]
    }).to_csv(OUT_DIR / "data_quality_summary.csv", index=False)
    fig_class_balance(y)
    fig_profiles(X, y)

    # ---- 5.2 Cross-validation set-up ----------------------------------------------- 
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, 
        n_repeats=N_REPEATS, 
        random_state=RANDOM_STATE
    )
    min_minority = min(
        int((y.iloc[tr] == 0).sum()) for tr, _ in cv.split(X, y))
    print(f"Smallest number of non-buyers in a training fold: {min_minority}")
    assert min_minority > K_NEIGHBORS, "Too few minority rows for SMOTE: lower K_NEIGHBORS"
    fingerprint = int(pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=False).sum() % 10**9)

    # ---- 5.3 Main evaluation -----------------------------------------------------------
    label, cw, sampler = VARIANTS["main"]
    print(f"\n=== Main evaluation: {label} ({N_SPLITS}x{N_REPEATS} CV) ===")
    results = {"main": run_cv(X, y, cv, "main", cw, sampler, fingerprint)}

    summary = summarize(results["main"])
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "cv_results_summary.csv")
    means(results["main"]).to_csv(OUT_DIR / "cv_results_means.csv")

    pairs = paired_comparison(results["main"])
    print(pairs.round(3).to_string(index=False))
    pairs.to_csv(OUT_DIR / "paired_comparison.csv", index=False)
    fig_cv_comparison(results["main"])

    # ---- 5.4 Out-of-fold curves, threshold analysis, confusion matrix -------------------
    print("\n=== Out-of-fold predictions ===")
    oof = oof_scores(X, y, fingerprint)
    table = fig_operating_points(oof, y)
    table.to_csv(OUT_DIR / "threshold_analysis_logistic_regression.csv", index=False)
    print(table.round(3).to_string(index=False))

    # ---- 5.5 Sensitivity analyses -------------------------------------------------------
    if RUN_ABLATIONS:
        for tag in ["smote_only", "smotenc", "weight_only"]:
            label, cw, sampler = VARIANTS[tag]
            print(f"\n=== Sensitivity: {label} ===")
            results[tag] = run_cv(X, y, cv, tag, cw, sampler, fingerprint)
            summarize(results[tag]).to_csv(OUT_DIR / f"cv_results_{tag}.csv")
        variant_means = {tag: means(r) for tag, r in results.items()}
        key = ["pr_auc", "mcc", "recall_minority", "precision_minority"]
        comparison = pd.concat({
            VARIANTS[t][0]: m[key].round(3) 
            for t, m in variant_means.items()
        }, axis=1)
        print(comparison.to_string())
        comparison.to_csv(OUT_DIR / "ablation_comparison.csv")
        fig_ablation(variant_means)

    # ---- 5.6 Interpretation graphs (built from cross-validation) ---------------------
    print("\n=== Interpretation ===")
    lr_smote = fold_models(X, y, make_models(True)["Logistic Regression"], "smote", cv)
    lr_plain = fold_models(X, y, make_models(True)["Logistic Regression"], None, cv)
    coef_smote = np.array([m.coef_[0] for m in lr_smote])
    coef_plain = np.array([m.coef_[0] for m in lr_plain])
    fig_lr_coefficients(coef_smote, coef_plain)

    dt_folds = fold_models(X, y, make_models(True)["Decision Tree"], "smote", cv)
    rf_folds = fold_models(X, y, make_models(True)["Random Forest"], "smote",
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE))
    imp_dt = np.array([m.feature_importances_ for m in dt_folds])
    imp_rf = np.array([m.feature_importances_ for m in rf_folds])
    fig_importances(imp_dt, imp_rf)
    
    pd.DataFrame({
        "feature": FEATURE_NAMES, 
        "dt_importance_mean": imp_dt.mean(0), 
        "dt_importance_std": imp_dt.std(0),
        "rf_importance_mean": imp_rf.mean(0), 
        "rf_importance_std": imp_rf.std(0)
        }).sort_values(
            "rf_importance_mean", 
            ascending=False
        ).to_csv(OUT_DIR / "tree_feature_importance.csv", index=False)

    # ---- 5.7 Final models fitted on ALL data (after the comparison) -----------------
    final_lr = make_pipe(make_models(True)["Logistic Regression"], "smote").fit(X, y)
    joblib.dump(final_lr, OUT_DIR / "final_logistic_regression_pipeline.joblib")

    coef_table = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "final_fit": final_lr.named_steps["clf"].coef_[0],
        "fold_mean": coef_smote.mean(0), "fold_std": coef_smote.std(0),
        "no_smote_fold_mean": coef_plain.mean(0),
        "no_smote_fold_std": coef_plain.std(0)
    }).sort_values("final_fit", ascending=False)
    print(coef_table.round(3).to_string(index=False))
    coef_table.to_csv(OUT_DIR / "lr_coefficients.csv", index=False)

    final_dt = make_pipe(make_models(True)["Decision Tree"], "smote").fit(X, y)
    fig_tree(final_dt.named_steps["clf"])

    # ---- 5.8 PCA on the real rows (unsupervised, exploratory) --------------------------
    Z = make_preprocessor().fit_transform(X)
    Z_std = StandardScaler().fit_transform(Z)

    scores, ev, loadings, buyers, non_buyers = pca_view(Z_std, y)
    print("Variance explained:", np.round(ev[:3] * 100, 1), 
          "| first three:", round(ev[:3].sum() * 100, 1), "%")
    print(loadings.abs().sort_values("PC1", ascending=False).round(3).to_string())

    loadings.to_csv(OUT_DIR / "pca_loadings.csv")
    fig_pca3d(scores, ev, buyers, non_buyers)
    print("\nDone. Figures in figures/, tables and the final model in results/.")

if __name__ == "__main__":
    main()