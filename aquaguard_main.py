"""
=============================================================================
AquaGuard — Predictive Water Pipe Failure Detection System
=============================================================================
Dataset     : Synthetic dataset generated to mirror real municipal water
              infrastructure data (pressure sensors + pipe metadata).
              Reference structure: Kaggle "Water Distribution System" datasets
              and UCI infrastructure failure datasets.
Pipeline    : Data Generation → EDA → Preprocessing → SMOTE →
              Feature Engineering (PCA) → ARIMA Time-Series →
              Multi-Model Training → SHAP Explainability → Export
=============================================================================
"""

import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import shap
import joblib
from statsmodels.tsa.arima.model import ARIMA

# REPLACE WITH:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
np.random.seed(42)

# =============================================================================
# STEP 1: SYNTHETIC DATASET GENERATION
# (mirrors real-world water infrastructure sensor + metadata data)
# =============================================================================
print("=" * 65)
print("STEP 1: GENERATING SYNTHETIC DATASET")
print("=" * 65)

N = 10_000   # total pipe segments

# Pipe metadata
pipe_age          = np.random.randint(1, 60, N)
pipe_diameter_mm  = np.random.choice([50, 100, 150, 200, 300, 400], N,
                                      p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])
material          = np.random.choice(["cast_iron", "ductile_iron", "PVC", "steel", "concrete"],
                                      N, p=[0.25, 0.20, 0.30, 0.15, 0.10])
soil_type         = np.random.choice(["clay", "sandy", "loam", "rocky"], N,
                                      p=[0.30, 0.25, 0.25, 0.20])
burial_depth_m    = np.round(np.random.uniform(0.5, 3.5, N), 2)
num_repairs       = np.random.poisson(lam=1.5, size=N)

# Sensor readings (daily averages over last 30 days)
avg_pressure_bar  = np.round(np.random.uniform(2.0, 8.0, N), 2)
pressure_variance = np.round(np.random.exponential(scale=0.5, size=N), 3)
avg_flow_lps      = np.round(np.random.uniform(5, 120, N), 2)
flow_variance     = np.round(np.random.exponential(scale=2.0, size=N), 3)
pressure_drop_pct = np.round(np.random.uniform(0, 15, N), 2)
temp_soil_c       = np.round(np.random.normal(18, 6, N), 1)

# Location
zone              = np.random.choice(["Zone_A", "Zone_B", "Zone_C", "Zone_D", "Zone_E"], N)
elevation_m       = np.round(np.random.uniform(10, 500, N), 1)

# Failure probability — engineered with domain logic
failure_score = (
    (pipe_age / 60) * 0.30 +
    (pressure_variance / 2.0) * 0.20 +
    (pressure_drop_pct / 15.0) * 0.15 +
    (num_repairs / 10.0) * 0.15 +
    (np.where(material == "cast_iron", 0.10, 0.0)) +
    (np.where(soil_type == "clay", 0.06, 0.0)) +
    np.random.uniform(0, 0.04, N)   # noise
)
failure_score = np.clip(failure_score, 0, 1)
failure       = (failure_score > 0.55).astype(int)

# Build DataFrame
df = pd.DataFrame({
    "pipe_id":           [f"P{str(i).zfill(5)}" for i in range(1, N + 1)],
    "pipe_age_yrs":      pipe_age,
    "diameter_mm":       pipe_diameter_mm,
    "material":          material,
    "soil_type":         soil_type,
    "burial_depth_m":    burial_depth_m,
    "num_prior_repairs": num_repairs,
    "avg_pressure_bar":  avg_pressure_bar,
    "pressure_variance": pressure_variance,
    "avg_flow_lps":      avg_flow_lps,
    "flow_variance":     flow_variance,
    "pressure_drop_pct": pressure_drop_pct,
    "soil_temp_c":       temp_soil_c,
    "elevation_m":       elevation_m,
    "zone":              zone,
    "failure":           failure
})

# Inject ~5% missing values (realistic sensor dropout)
for col in ["avg_pressure_bar", "pressure_variance", "avg_flow_lps",
            "pressure_drop_pct", "soil_temp_c"]:
    mask = np.random.rand(N) < 0.05
    df.loc[mask, col] = np.nan

# Inject ~2% outliers in pressure
outlier_idx = np.random.choice(N, int(N * 0.02), replace=False)
df.loc[outlier_idx, "avg_pressure_bar"] = np.random.uniform(12, 20, len(outlier_idx))

df.to_csv(os.path.join(BASE_DIR, "outputs", "aquaguard_dataset.csv"), index=False)
print(f"  Dataset shape     : {df.shape}")
print(f"  Failure cases     : {df['failure'].sum()} ({df['failure'].mean()*100:.1f}%)")
print(f"  Non-failure cases : {(df['failure']==0).sum()} ({(1-df['failure'].mean())*100:.1f}%)")
print(f"  Missing values    : {df.isnull().sum().sum()} total")

# =============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 65)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print("\n--- Basic Statistics ---")
print(df.describe().round(3).to_string())

print("\n--- Missing Values per Column ---")
mv = df.isnull().sum()
print(mv[mv > 0].to_string())

print("\n--- Class Distribution ---")
print(df["failure"].value_counts())
print(f"  Imbalance ratio : {(df['failure']==0).sum() / df['failure'].sum():.1f} : 1")

# EDA Plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("AquaGuard — EDA: Feature Distributions", fontsize=15, fontweight="bold")

axes[0, 0].hist(df["pipe_age_yrs"], bins=30, color="#2196F3", edgecolor="white")
axes[0, 0].set_title("Pipe Age Distribution"); axes[0, 0].set_xlabel("Age (years)")

sns.countplot(ax=axes[0, 1], x="material", data=df, palette="Set2",
              order=df["material"].value_counts().index)
axes[0, 1].set_title("Material Types"); axes[0, 1].tick_params(axis="x", rotation=25)

axes[0, 2].hist(df["avg_pressure_bar"].dropna(), bins=30, color="#FF5722", edgecolor="white")
axes[0, 2].set_title("Avg Pressure (bar)"); axes[0, 2].set_xlabel("Pressure (bar)")

axes[1, 0].bar(["No Failure (0)", "Failure (1)"],
               df["failure"].value_counts().sort_index(),
               color=["#4CAF50", "#F44336"])
axes[1, 0].set_title("Class Imbalance"); axes[1, 0].set_ylabel("Count")

df.boxplot(column="pipe_age_yrs", by="failure", ax=axes[1, 1])
axes[1, 1].set_title("Pipe Age vs Failure"); axes[1, 1].set_xlabel("Failure Label")

df.boxplot(column="pressure_drop_pct", by="failure", ax=axes[1, 2])
axes[1, 2].set_title("Pressure Drop % vs Failure"); axes[1, 2].set_xlabel("Failure Label")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "eda_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  EDA plot saved: eda_distributions.png")

# Correlation heatmap
num_cols = ["pipe_age_yrs", "diameter_mm", "burial_depth_m", "num_prior_repairs",
            "avg_pressure_bar", "pressure_variance", "avg_flow_lps",
            "flow_variance", "pressure_drop_pct", "soil_temp_c", "elevation_m", "failure"]
fig, ax = plt.subplots(figsize=(12, 9))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Correlation heatmap saved")

# =============================================================================
# STEP 3: PREPROCESSING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 3: PREPROCESSING")
print("=" * 65)

df_clean = df.copy()

# 3a. Missing value imputation
num_features = ["avg_pressure_bar", "pressure_variance", "avg_flow_lps",
                "pressure_drop_pct", "soil_temp_c"]
for col in num_features:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)
print(f"  Missing values after imputation: {df_clean.isnull().sum().sum()}")

# 3b. Outlier treatment (IQR capping)
for col in ["avg_pressure_bar", "avg_flow_lps"]:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    clipped = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    df_clean[col] = df_clean[col].clip(lower, upper)
    print(f"  Outliers clipped in '{col}': {clipped}")

# 3c. Encode categoricals
le_material  = LabelEncoder()
le_soil      = LabelEncoder()
le_zone      = LabelEncoder()
df_clean["material_enc"]  = le_material.fit_transform(df_clean["material"])
df_clean["soil_type_enc"] = le_soil.fit_transform(df_clean["soil_type"])
df_clean["zone_enc"]      = le_zone.fit_transform(df_clean["zone"])

# Features & target
feature_cols = [
    "pipe_age_yrs", "diameter_mm", "burial_depth_m", "num_prior_repairs",
    "avg_pressure_bar", "pressure_variance", "avg_flow_lps", "flow_variance",
    "pressure_drop_pct", "soil_temp_c", "elevation_m",
    "material_enc", "soil_type_enc", "zone_enc"
]
X = df_clean[feature_cols]
y = df_clean["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Train failure rate: {y_train.mean()*100:.1f}%")

# 3d. Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 3e. SMOTE — handle class imbalance
print(f"\n  Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
print(f"  After  SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# =============================================================================
# STEP 4: FEATURE ENGINEERING — PCA
# =============================================================================
print("\n" + "=" * 65)
print("STEP 4: FEATURE ENGINEERING — PCA")
print("=" * 65)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_res)
X_test_pca  = pca.transform(X_test_sc)
print(f"  Original features  : {X_train_res.shape[1]}")
print(f"  PCA components     : {pca.n_components_} (95% variance)")
print(f"  Explained variance : {pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}%")

# PCA Scree Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.cumsum(pca.explained_variance_ratio_) * 100, "bo-", linewidth=2)
ax.axhline(95, color="red", linestyle="--", label="95% threshold")
ax.set_xlabel("Number of Components"); ax.set_ylabel("Cumulative Explained Variance (%)")
ax.set_title("PCA — Cumulative Explained Variance"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "pca_scree.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  PCA scree plot saved")

# =============================================================================
# STEP 5: ARIMA TIME SERIES — PRESSURE FORECASTING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 5: ARIMA — PRESSURE TREND FORECASTING")
print("=" * 65)

# Simulate 60-day daily pressure readings for a single zone (Zone_A high-risk)
np.random.seed(99)
days     = 60
baseline = 5.5
trend    = np.linspace(0, -0.8, days)          # slow pressure decline → leak signal
noise    = np.random.normal(0, 0.15, days)
seasonal = 0.3 * np.sin(2 * np.pi * np.arange(days) / 7)  # weekly pattern
pressure_series = baseline + trend + noise + seasonal

arima_model = ARIMA(pressure_series, order=(2, 1, 2))
arima_fit   = arima_model.fit()
forecast    = arima_fit.forecast(steps=14)
conf_int    = arima_fit.get_forecast(steps=14).conf_int()

print(f"  ARIMA(2,1,2) AIC   : {arima_fit.aic:.2f}")
print(f"  14-day forecast    : {np.round(forecast, 3).tolist()}")
alert_threshold = 4.8
if forecast.min() < alert_threshold:
    print(f"  ⚠ ALERT: Forecasted pressure drops below {alert_threshold} bar"
          f" on day {np.argmin(forecast)+1} → Potential leak in Zone_A!")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(days), pressure_series, "b-", label="Historical Pressure", linewidth=1.5)
forecast_x = range(days, days + 14)
ax.plot(forecast_x, forecast, "r--", label="ARIMA Forecast (14 days)", linewidth=2)
ax.fill_between(forecast_x, conf_int[:, 0], conf_int[:, 1],
                alpha=0.2, color="red", label="95% Confidence Interval")
ax.axhline(alert_threshold, color="orange", linestyle=":", label=f"Alert threshold ({alert_threshold} bar)")
ax.set_xlabel("Day"); ax.set_ylabel("Pressure (bar)")
ax.set_title("Zone_A — ARIMA Pressure Forecasting (Leak Early Warning)", fontsize=12)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "arima_forecast.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ARIMA forecast plot saved")

# =============================================================================
# STEP 6: MODEL TRAINING & COMPARISON
# =============================================================================
print("\n" + "=" * 65)
print("STEP 6: MODEL TRAINING & COMPARISON")
print("=" * 65)

# We train on SMOTE resampled SCALED data, test on held-out scaled data
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12,
                                                  class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                      max_depth=5, random_state=42),
    "XGBoost":             xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                              scale_pos_weight=10, use_label_encoder=False,
                                              eval_metric="logloss", random_state=42,
                                              tree_method="hist")
}

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    # CV on resampled training data
    cv_roc   = cross_val_score(model, X_train_res, y_train_res, cv=cv,
                               scoring="roc_auc", n_jobs=-1)
    cv_f1    = cross_val_score(model, X_train_res, y_train_res, cv=cv,
                               scoring="f1", n_jobs=-1)
    # Fit and evaluate on held-out test
    model.fit(X_train_res, y_train_res)
    y_prob   = model.predict_proba(X_test_sc)[:, 1]
    y_pred   = model.predict(X_test_sc)
    test_roc = roc_auc_score(y_test, y_prob)
    test_f1  = f1_score(y_test, y_pred)
    test_ap  = average_precision_score(y_test, y_prob)
    results[name] = {
        "CV_ROC_AUC": f"{cv_roc.mean():.4f} ± {cv_roc.std():.4f}",
        "CV_F1":      f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}",
        "Test_ROC_AUC": round(test_roc, 4),
        "Test_F1":      round(test_f1, 4),
        "Test_AP":      round(test_ap, 4),
        "model_obj":    model,
        "y_prob":       y_prob,
        "y_pred":       y_pred
    }
    print(f"\n  {name}")
    print(f"    CV  ROC-AUC : {results[name]['CV_ROC_AUC']}")
    print(f"    CV  F1      : {results[name]['CV_F1']}")
    print(f"    Test ROC-AUC: {results[name]['Test_ROC_AUC']}")
    print(f"    Test F1     : {results[name]['Test_F1']}")
    print(f"    Test Avg-P  : {results[name]['Test_AP']}")

# Best model → XGBoost
best_name  = "XGBoost"
best_model = results[best_name]["model_obj"]

# Classification report
print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, results[best_name]["y_pred"],
                             target_names=["No Failure", "Failure"]))

# =============================================================================
# STEP 7: VISUALISATIONS — ROC, PR, Confusion Matrix
# =============================================================================
print("\n" + "=" * 65)
print("STEP 7: EVALUATION VISUALISATIONS")
print("=" * 65)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curves — all models
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['Test_ROC_AUC']})", linewidth=2)
axes[0].plot([0,1],[0,1],"k--"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curves — All Models"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# Precision-Recall Curve (XGBoost)
prec, rec, _ = precision_recall_curve(y_test, results[best_name]["y_prob"])
axes[1].plot(rec, prec, "b-", linewidth=2)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title(f"Precision-Recall — {best_name}"); axes[1].grid(alpha=0.3)

# Confusion Matrix (XGBoost)
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"])
axes[2].set_title(f"Confusion Matrix — {best_name}")
axes[2].set_ylabel("Actual"); axes[2].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "model_evaluation.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Model evaluation plots saved")

# =============================================================================
# STEP 8: SHAP EXPLAINABILITY
# =============================================================================
print("\n" + "=" * 65)
print("STEP 8: SHAP EXPLAINABILITY")
print("=" * 65)

explainer   = shap.TreeExplainer(best_model)
# Use full unmodified scaled test set
shap_values = explainer.shap_values(X_test_sc)

# Handle XGBoost returning 2D
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

print("  SHAP values computed successfully")

# SHAP Summary Plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv, X_test_sc, feature_names=feature_cols,
                  show=False, plot_size=None)
plt.title("SHAP Summary — XGBoost Feature Importance", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

# SHAP Bar Plot
mean_shap   = np.abs(sv).mean(axis=0)
shap_df     = pd.DataFrame({"feature": feature_cols, "mean_shap": mean_shap})
shap_df     = shap_df.sort_values("mean_shap", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(shap_df["feature"], shap_df["mean_shap"], color="#FF7043")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("XGBoost — Global Feature Importance (SHAP)", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  SHAP plots saved")

# Top contributors
top3 = shap_df.nlargest(3, "mean_shap")["feature"].tolist()
print(f"  Top 3 features (SHAP): {top3}")

# SHAP for single high-risk pipe
high_risk_idx = np.argmax(results[best_name]["y_prob"])
fig, ax = plt.subplots(figsize=(10, 4))
shap.waterfall_plot(
    shap.Explanation(values=sv[high_risk_idx],
                     base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                                 else explainer.expected_value[1],
                     feature_names=feature_cols),
    show=False
)
plt.title(f"SHAP Waterfall — Highest-Risk Pipe (Prob={results[best_name]['y_prob'][high_risk_idx]:.2%})",
          fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "plots", "shap_waterfall.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  SHAP waterfall plot saved")

# =============================================================================
# STEP 9: MODEL EXPORT
# =============================================================================
print("\n" + "=" * 65)
print("STEP 9: MODEL EXPORT (Joblib)")
print("=" * 65)

joblib.dump(best_model, os.path.join(BASE_DIR, "models", "xgboost_aquaguard.pkl"))
joblib.dump(scaler,     os.path.join(BASE_DIR, "models", "scaler.pkl"))
joblib.dump(pca,        os.path.join(BASE_DIR, "models", "pca.pkl"))
meta = {
    "model":          "XGBoost",
    "features":       feature_cols,
    "test_roc_auc":   results[best_name]["Test_ROC_AUC"],
    "test_f1":        results[best_name]["Test_F1"],
    "top_shap_feats": top3
}
with open(os.path.join(BASE_DIR, "models", "model_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("  Saved: xgboost_aquaguard.pkl")
print("  Saved: scaler.pkl")
print("  Saved: pca.pkl")
print("  Saved: model_metadata.json")

# =============================================================================
# STEP 10: RESULTS SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 65)
print("STEP 10: FINAL RESULTS SUMMARY")
print("=" * 65)

summary_rows = []
for name, res in results.items():
    summary_rows.append({
        "Model":        name,
        "CV ROC-AUC":  res["CV_ROC_AUC"],
        "Test ROC-AUC": res["Test_ROC_AUC"],
        "Test F1":      res["Test_F1"],
        "Test Avg-P":   res["Test_AP"]
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(BASE_DIR, "outputs", "model_comparison.csv"), index=False)

print(f"\n✅  AquaGuard pipeline complete. All outputs in {BASE_DIR}")
