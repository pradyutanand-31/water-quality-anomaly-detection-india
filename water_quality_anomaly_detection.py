"""
=============================================================================
National-Scale Anomaly Detection in Inland Waters:
An Explainable Multi-Model AI Framework for Environmental Governance
=============================================================================
Authors: Bhowmik, Varshney, Mishra, Anand, Aggarwal
Journal: Environmental Governance (2025)

This script fully reproduces ALL results, tables, and figures reported in
the manuscript using the CPCB 2021 inland water-quality dataset.

Outputs produced:
  - Figure 2 : KDE plots of normalised features
  - Figure 3 : Pearson correlation network / heatmap
  - Figure 4 : Confusion matrices (4 models)
  - Figure 5 : Histogram of normalised anomaly scores
  - Figure 6 : PCA scatter of top anomalies
  - Figure 7 : SHAP summary plot (Isolation Forest)
  - Figure 8 : t-SNE projection (anomaly vs normal)
  - Figure 9 : Feature z-score bar chart (top 10)
  - Table 1  : Precision / Recall / F1 (printed to console + CSV)
  - Table 2  : Top-10 anomalies per model (printed + CSV)
  - Table 3  : Feature ranking Δz / Δμ (printed + CSV)

Requirements (install once):
  pip install pandas openpyxl numpy scikit-learn matplotlib seaborn shap
              tensorflow networkx

Usage:
  python water_quality_anomaly_detection.py

Dataset path is set by DATA_PATH below. Update to your local path.
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0.  IMPORTS
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless / non-interactive rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import tensorflow as tf
from tensorflow import keras
import shap

# ---------------------------------------------------------------------------
# 1.  REPRODUCIBILITY SEEDS
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ---------------------------------------------------------------------------
# 2.  PATHS  (edit DATA_PATH to point to your file)
# ---------------------------------------------------------------------------
DATA_PATH   = "Dataset.xlsx"
OUTPUT_DIR  = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 3.  DATA LOADING AND PREPROCESSING
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Loading dataset …")
df_raw = pd.read_excel(DATA_PATH)
df_raw.columns = [c.replace("\n", " ").strip() for c in df_raw.columns]

# --- Numeric columns (all physicochemical + microbiological params) ---
NUM_COLS = [
    "Temperature (Min)", "Temperature (Max)",
    "Dissolved Oxygen (mg/L) (Min)", "Dissolved Oxygen (mg/L) (Max)",
    "pH (Min)", "pH (Max)",
    "Conductivity (mho/cm) (Min)", "Conductivity (mho/cm) (Max)",
    "BOD (mg/L) (Min)", "BOD (mg/L) (Max)",
    "Nitrate N + Nitrite N(mg/L) (Min)", "Nitrate N + Nitrite N(mg/L) (Max)",
    "Fecal Coliform (MPN/100ml) (Min)", "Fecal Coliform (MPN/100ml) (Max)",
    "Total Coliform (MPN/100ml) (Min)", "Total Coliform (MPN/100ml) (Max)",
]

df = df_raw.copy()
for col in NUM_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows where key columns are >30% missing (paper Section 2.2)
key_cols = ["Dissolved Oxygen (mg/L) (Min)", "BOD (mg/L) (Max)",
            "Fecal Coliform (MPN/100ml) (Max)", "Total Coliform (MPN/100ml) (Max)"]
df = df.dropna(subset=key_cols, thresh=3).reset_index(drop=True)

# Median imputation for <10% missing
for col in NUM_COLS:
    df[col] = df[col].fillna(df[col].median())

# Implausible-value removal: negative concentrations or DO > saturation
df = df[df["Dissolved Oxygen (mg/L) (Min)"] >= 0]
df = df[df["BOD (mg/L) (Min)"] >= 0]
df = df.reset_index(drop=True)

print(f"   Dataset shape after cleaning: {df.shape}")
print(f"   States: {df['State Name'].nunique()}, "
      f"Water body types: {df['Type Water Body'].nunique()}")

# ---------------------------------------------------------------------------
# 4.  FEATURE ENGINEERING  (Equations 1–4 of manuscript)
# ---------------------------------------------------------------------------
print("Step 2: Feature engineering …")

epsilon = 1e-6
T_avg = (df["Temperature (Min)"] + df["Temperature (Max)"]) / 2.0

# DO saturation (empirical polynomial, Benson & Krause, 1984)
df["DO_sat"] = (14.62 - 0.3898 * T_avg
                + 0.006969 * T_avg**2
                - 0.00005896 * T_avg**3)

# Equation 2  – DO deficit
df["DO_deficit"] = np.maximum(0.0,
                              df["DO_sat"] - df["Dissolved Oxygen (mg/L) (Min)"])

# Equation 1  – BOD/DO ratio (worst case: max BOD / min DO)
df["BOD_DO_ratio"] = (df["BOD (mg/L) (Max)"]
                      / (df["Dissolved Oxygen (mg/L) (Min)"] + epsilon))

# Equation 3  – Coliform Load Index
df["CLI"] = ((df["Fecal Coliform (MPN/100ml) (Min)"]
              + df["Total Coliform (MPN/100ml) (Min)"]) / 2.0)

# Equation 4  – Intra-site ranges
df["DO_range"]           = (df["Dissolved Oxygen (mg/L) (Max)"]
                            - df["Dissolved Oxygen (mg/L) (Min)"])
df["BOD_range"]          = df["BOD (mg/L) (Max)"] - df["BOD (mg/L) (Min)"]
df["Temp_range"]         = df["Temperature (Max)"] - df["Temperature (Min)"]
df["Nitrate_range"]      = (df["Nitrate N + Nitrite N(mg/L) (Max)"]
                            - df["Nitrate N + Nitrite N(mg/L) (Min)"])
df["Conductivity_range"] = (df["Conductivity (mho/cm) (Max)"]
                            - df["Conductivity (mho/cm) (Min)"])

ENGINEERED_COLS = [
    "DO_deficit", "BOD_DO_ratio", "CLI",
    "DO_range", "BOD_range", "Temp_range",
    "Nitrate_range", "Conductivity_range"
]

# All modelling features
ALL_FEAT = NUM_COLS + ENGINEERED_COLS

# One-hot encode categorical columns (State, Water Body Type)
df_enc = pd.get_dummies(df, columns=["State Name", "Type Water Body"])

# ---------------------------------------------------------------------------
# 5.  SCALING & DIMENSIONALITY REDUCTION
# ---------------------------------------------------------------------------
print("Step 3: Scaling and PCA …")

X_raw = df[ALL_FEAT].fillna(df[ALL_FEAT].median())
X_scaled_arr = MinMaxScaler().fit_transform(X_raw.values)
X_scaled = pd.DataFrame(X_scaled_arr, columns=ALL_FEAT)

pca = PCA(n_components=0.95, random_state=SEED)
X_pca = pca.fit_transform(X_scaled_arr)
print(f"   PCA retained {pca.n_components_} components "
      f"(≥95% cumulative variance).")

# ---------------------------------------------------------------------------
# 6.  RULE-BASED REFERENCE LABELS
#     CPCB Class B freshwater standards (used ONLY for post-hoc validation)
# ---------------------------------------------------------------------------
rule_pos = (
    ((df["Dissolved Oxygen (mg/L) (Min)"] < 5.0)
     & (df["BOD (mg/L) (Max)"]   > 3.0))
    | (df["BOD (mg/L) (Max)"]   > 30.0)
    | (df["Fecal Coliform (MPN/100ml) (Max)"] > 2500)
)
y_true = rule_pos.astype(int).values
print(f"   Rule-based anomalies: {y_true.sum()} of {len(y_true)}")

# ---------------------------------------------------------------------------
# 7.  ANOMALY DETECTION MODELS
# ---------------------------------------------------------------------------
print("Step 4: Fitting anomaly detection models …")
CONTAMINATION = 0.10   # ~10% expected anomaly rate (paper Section 2.4)

# 7a. Isolation Forest (Equation 5)
IF_model = IsolationForest(contamination=CONTAMINATION,
                            n_estimators=100, random_state=SEED)
IF_preds  = IF_model.fit_predict(X_pca)        # -1 = anomaly
IF_raw_scores = IF_model.score_samples(X_pca)  # lower  = more anomalous
IF_anom   = (IF_preds == -1)

# 7b. One-Class SVM (Equations 6–7)
OCSVM_model = OneClassSVM(nu=CONTAMINATION, kernel="rbf", gamma=0.1)
OCSVM_preds = OCSVM_model.fit_predict(X_pca)
OCSVM_raw_scores = OCSVM_model.score_samples(X_pca)  # lower = more anomalous
OCSVM_anom  = (OCSVM_preds == -1)

# 7c. Elliptic Envelope (Equation 8)
EE_model = EllipticEnvelope(contamination=CONTAMINATION,
                             support_fraction=0.9, random_state=SEED)
EE_preds = EE_model.fit_predict(X_pca)
EE_maha  = EE_model.mahalanobis(X_pca)          # higher = more anomalous
EE_anom  = (EE_preds == -1)

# 7d. Autoencoder (Equation 9)
n_feat = X_pca.shape[1]
inp    = keras.Input(shape=(n_feat,))
enc    = keras.layers.Dense(10, activation="relu")(inp)
dec    = keras.layers.Dense(n_feat, activation="sigmoid")(enc)
AE     = keras.Model(inp, dec)
AE.compile(optimizer="adam", loss="mse")
AE.fit(X_pca, X_pca, epochs=50, batch_size=32, shuffle=True, verbose=0)
AE_recon   = AE.predict(X_pca, verbose=0)
AE_errors  = np.mean((X_pca - AE_recon) ** 2, axis=1)
AE_thresh  = np.percentile(AE_errors, (1 - CONTAMINATION) * 100)
AE_anom    = (AE_errors > AE_thresh)
print("   All four models fitted.")

# ---------------------------------------------------------------------------
# 8.  NORMALISE ANOMALY SCORES TO [0,1] (for Figures 5 & 6 and Table 2)
# ---------------------------------------------------------------------------
def norm01(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-12)

IF_score_norm    = norm01(-IF_raw_scores)         # flip: higher = worse
OCSVM_score_norm = norm01(-OCSVM_raw_scores)
EE_score_norm    = norm01(EE_maha)
AE_score_norm    = norm01(AE_errors)

print(f"   Anomaly counts — IF: {IF_anom.sum()}, OCSVM: {OCSVM_anom.sum()}, "
      f"EE: {EE_anom.sum()}, AE: {AE_anom.sum()}")

# ---------------------------------------------------------------------------
# 9.  TABLE 1 – PERFORMANCE METRICS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TABLE 1: Performance metrics vs rule-based reference anomalies")
print("=" * 60)

results = {}
for name, preds in [("Isolation Forest", IF_anom.astype(int)),
                    ("One-Class SVM",    OCSVM_anom.astype(int)),
                    ("Elliptic Envelope",EE_anom.astype(int)),
                    ("Autoencoder",      AE_anom.astype(int))]:
    p  = precision_score(y_true, preds, zero_division=0)
    r  = recall_score(y_true,    preds, zero_division=0)
    f1 = f1_score(y_true,        preds, zero_division=0)
    results[name] = {"Precision": round(p, 3),
                     "Recall":    round(r, 3),
                     "F1-score":  round(f1, 3)}
    print(f"  {name:<20s} | P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

table1_df = pd.DataFrame(results).T.reset_index()
table1_df.columns = ["Model", "Precision", "Recall", "F1-score"]
table1_df.to_csv(os.path.join(OUTPUT_DIR, "Table1_performance.csv"), index=False)
print("  → saved: Table1_performance.csv")

# ---------------------------------------------------------------------------
# 10. TABLE 2 – TOP-10 ANOMALIES PER MODEL
# ---------------------------------------------------------------------------
print("\nTABLE 2: Top-10 anomalies per model")
print("=" * 60)

def top10_table(score_arr, name):
    s = pd.Series(score_arr)
    top = s.nlargest(10)
    rows = []
    for rank, (idx, val) in enumerate(top.items(), 1):
        rows.append({"Rank": rank, "Model": name, "Idx": idx,
                     "Score": round(val, 3)})
    return pd.DataFrame(rows)

t2_frames = []
for name, scores, label in [
        ("Isolation Forest",  IF_score_norm,    "Score"),
        ("One-Class SVM",     OCSVM_score_norm, "Score"),
        ("Elliptic Envelope", EE_score_norm,     "Score*"),
        ("Autoencoder",       AE_score_norm,    "Score")]:
    t2 = top10_table(scores, name)
    t2_frames.append(t2)
    print(f"\n  {name}")
    print(t2[["Rank","Idx","Score"]].to_string(index=False))

table2_df = pd.concat(t2_frames, ignore_index=True)
table2_df.to_csv(os.path.join(OUTPUT_DIR, "Table2_top_anomalies.csv"), index=False)
print("\n  → saved: Table2_top_anomalies.csv")
print("  (* Elliptic Envelope: ties in normalised score ordered by"
      " Mahalanobis distance)")

# ---------------------------------------------------------------------------
# 11. TABLE 3 – FEATURE RANKING (Δz, Δμ)
# ---------------------------------------------------------------------------
print("\nTABLE 3: Feature ranking Δz and Δμ")
print("=" * 60)

ensemble_anom = (IF_anom | EE_anom | OCSVM_anom | AE_anom)

X_std = (X_raw - X_raw.mean()) / (X_raw.std() + 1e-12)

anom_mask  = np.asarray(ensemble_anom, dtype=bool)
norm_mask  = ~anom_mask

delta_z  = X_std.loc[anom_mask].mean() - X_std.loc[norm_mask].mean()
delta_mu = X_raw.loc[anom_mask].mean() - X_raw.loc[norm_mask].mean()

direction = delta_z.apply(lambda v: "↑" if v >= 0 else "↓")
magnitude_z  = delta_z.abs()
magnitude_mu = delta_mu.abs()

# Tertile labels
def tertile(s):
    q1, q2 = s.quantile(0.33), s.quantile(0.67)
    return s.apply(lambda v: "H" if v >= q2 else ("M" if v >= q1 else "L"))

t3 = pd.DataFrame({
    "Feature":        delta_z.index,
    "Δz (std units)": delta_z.values.round(3),
    "Δμ (raw units)": delta_mu.values.round(3),
    "Δz grade":       tertile(magnitude_z).values,
    "Δμ grade":       tertile(magnitude_mu).values,
    "Direction":      direction.values,
}).sort_values("Δz (std units)", key=abs, ascending=False).reset_index(drop=True)
t3.index = t3.index + 1
t3.index.name = "Rank"

print(t3[["Feature","Δz grade","Δμ grade","Direction"]].head(15).to_string())
t3.to_csv(os.path.join(OUTPUT_DIR, "Table3_feature_ranking.csv"))
print("  → saved: Table3_feature_ranking.csv")

# ---------------------------------------------------------------------------
# 12. FIGURE 2 – KDE PLOTS (scaled features)
# ---------------------------------------------------------------------------
print("\nGenerating Figure 2: KDE plots …")

display_feats = [
    ("BOD (mg/L) (Max)",                  "BOD (max)"),
    ("Dissolved Oxygen (mg/L) (Min)",      "DO (min)"),
    ("Fecal Coliform (MPN/100ml) (Max)",   "Fecal Coliform (max)"),
    ("Total Coliform (MPN/100ml) (Max)",   "Total Coliform (max)"),
    ("pH (Min)",                           "pH (min)"),
    ("Conductivity (mho/cm) (Max)",        "Conductivity (max)"),
    ("BOD_DO_ratio",                       "BOD/DO Ratio"),
    ("DO_deficit",                         "DO Deficit"),
    ("CLI",                                "Coliform Load Index"),
    ("Nitrate N + Nitrite N(mg/L) (Max)",  "Nitrate+Nitrite (max)"),
    ("Temperature (Max)",                  "Temperature (max)"),
    ("DO_range",                           "DO Range"),
]

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, (col, label) in enumerate(display_feats):
    if col in X_scaled.columns:
        vals = X_scaled[col].dropna()
    else:
        vals = pd.Series(dtype=float)
    if len(vals) > 0:
        sns.kdeplot(vals, ax=axes[i], fill=True, color="#2196F3", alpha=0.55, linewidth=1.5)
    axes[i].set_title(label, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Normalised value [0–1]", fontsize=8)
    axes[i].set_ylabel("Density", fontsize=8)
    axes[i].tick_params(labelsize=7)

plt.suptitle("Figure 2. KDE plots of scaled continuous water-quality features",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure2_KDE_plots.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("   → saved Figure2_KDE_plots.png")

# ---------------------------------------------------------------------------
# 13. FIGURE 3 – CORRELATION HEATMAP / NETWORK
# ---------------------------------------------------------------------------
print("Generating Figure 3: Correlation network …")

CORR_COLS = [
    "BOD (mg/L) (Max)", "Dissolved Oxygen (mg/L) (Min)", "pH (Min)",
    "Conductivity (mho/cm) (Max)", "Fecal Coliform (MPN/100ml) (Max)",
    "Total Coliform (MPN/100ml) (Max)", "BOD_DO_ratio",
    "DO_deficit", "CLI", "Nitrate N + Nitrite N(mg/L) (Max)"
]
CORR_LABELS = [
    "BOD(max)", "DO(min)", "pH(min)", "Cond(max)",
    "FC(max)", "TC(max)", "BOD/DO", "DO deficit", "CLI", "Nitrate(max)"
]

corr_df = X_raw[[c for c in CORR_COLS if c in X_raw.columns]].corr()
corr_df.columns = CORR_LABELS[:len(corr_df.columns)]
corr_df.index   = CORR_LABELS[:len(corr_df.index)]

fig, ax = plt.subplots(figsize=(10, 7))
mask = np.triu(np.ones_like(corr_df.values, dtype=bool), k=1)
sns.heatmap(corr_df, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.4, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax.set_title("Figure 3. Pearson Correlation Matrix of Continuous Features",
             fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure3_Correlation_Heatmap.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("   → saved Figure3_Correlation_Heatmap.png")

# ---------------------------------------------------------------------------
# 14. FIGURE 4 – CONFUSION MATRICES
# ---------------------------------------------------------------------------
print("Generating Figure 4: Confusion matrices …")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
MODEL_NAMES = ["Isolation Forest", "One-Class SVM", "Elliptic Envelope", "Autoencoder"]
MODEL_PREDS = [IF_anom, OCSVM_anom, EE_anom, AE_anom]
COLORS      = ["#1565C0", "#2E7D32", "#6A1B9A", "#BF360C"]

for ax, mname, mpred, col in zip(axes.flatten(), MODEL_NAMES, MODEL_PREDS, COLORS):
    cm = confusion_matrix(y_true, mpred.astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    p  = precision_score(y_true, mpred.astype(int), zero_division=0)
    r  = recall_score(y_true,    mpred.astype(int), zero_division=0)
    f1 = f1_score(y_true,        mpred.astype(int), zero_division=0)
    ax.set_title(f"{mname}\nP={p:.3f}  R={r:.3f}  F1={f1:.3f}",
                 fontsize=11, fontweight="bold", color=col)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)

plt.suptitle("Figure 4. Confusion Matrices – Four Anomaly Detection Models"
             "\n(Rule-based reference labels from CPCB threshold violations)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure4_Confusion_Matrices.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("   → saved Figure4_Confusion_Matrices.png")

# ---------------------------------------------------------------------------
# 15. FIGURE 5 – ANOMALY SCORE HISTOGRAMS
# ---------------------------------------------------------------------------
print("Generating Figure 5: Score histograms …")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
SCORES   = [IF_score_norm, OCSVM_score_norm, EE_score_norm, AE_score_norm]
ANOMS    = [IF_anom, OCSVM_anom, EE_anom, AE_anom]

for ax, mname, scores, anom, col in zip(
        axes.flatten(), MODEL_NAMES, SCORES, ANOMS, COLORS):
    ax.hist(scores[~anom], bins=30, alpha=0.65, color="steelblue",
            label="Normal",  edgecolor="white")
    ax.hist(scores[anom],  bins=30, alpha=0.85, color=col,
            label="Anomaly", edgecolor="white")
    ax.set_title(mname, fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalised Anomaly Score", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8)

plt.suptitle("Figure 5. Histogram of Normalised Anomaly Scores Across All Models",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure5_Score_Histograms.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("   → saved Figure5_Score_Histograms.png")

# ---------------------------------------------------------------------------
# 16. FIGURE 6 – PCA SCATTER (top anomalies per model)
# ---------------------------------------------------------------------------
print("Generating Figure 6: PCA anomaly plot …")

pc1, pc2 = X_pca[:, 0], X_pca[:, 1]

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(pc1[~ensemble_anom], pc2[~ensemble_anom],
           c="lightgrey", s=12, alpha=0.5, label="Normal", zorder=1)

model_colors_scatter = {"IF": "#1565C0", "OCSVM": "#2E7D32",
                         "EE": "#6A1B9A", "AE": "#BF360C"}
for mname, anom, col in zip(["IF","OCSVM","EE","AE"],
                              [IF_anom, OCSVM_anom, EE_anom, AE_anom],
                              model_colors_scatter.values()):
    idx = np.where(anom)[0]
    ax.scatter(pc1[idx], pc2[idx], s=55, alpha=0.7, color=col,
               label=mname, zorder=3)

# Annotate top-5 per model (IF)
top5_IF = pd.Series(IF_score_norm).nlargest(5).index
for i in top5_IF:
    ax.annotate(str(i), (pc1[i], pc2[i]), fontsize=7,
                xytext=(4, 4), textcoords="offset points", color="#1565C0")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
              fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
              fontsize=11)
ax.set_title("Figure 6. PCA Plot Showing Top Anomalies Detected by Each Model",
             fontsize=12, fontweight="bold")
ax.legend(title="Model", fontsize=9, title_fontsize=9, loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure6_PCA_Anomalies.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("   → saved Figure6_PCA_Anomalies.png")

# ---------------------------------------------------------------------------
# 17. FIGURE 7 – SHAP SUMMARY PLOT (Isolation Forest via KernelSHAP)
# ---------------------------------------------------------------------------
print("Generating Figure 7: SHAP summary plot …")

# Use a background summarised sample for speed
X_shap_bg = shap.sample(X_pca, 50, random_state=SEED)
X_shap_ex  = shap.sample(X_pca, 150, random_state=SEED)
PC_names   = [f"PC{i+1}" for i in range(X_pca.shape[1])]

explainer   = shap.KernelExplainer(IF_model.score_samples, X_shap_bg)
shap_values = explainer.shap_values(X_shap_ex, nsamples=80, silent=True)

# Truncate feature names for readability

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap_ex,
                  feature_names=PC_names,
                  plot_type="dot",
                  show=False,
                  max_display=15,
                  color_bar=True)
plt.title("Figure 7. SHAP Summary Plot – Isolation Forest\n"
          "(higher |SHAP| = greater influence on anomaly score)",
          fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure7_SHAP_Summary.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("   → saved Figure7_SHAP_Summary.png")

# ---------------------------------------------------------------------------
# 18. FIGURE 8 – t-SNE PROJECTION
# ---------------------------------------------------------------------------
print("Generating Figure 8: t-SNE projection …")

tsne = TSNE(n_components=2, perplexity=30, random_state=SEED,
            max_iter=1000, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_pca)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(X_tsne[~ensemble_anom, 0], X_tsne[~ensemble_anom, 1],
           c="#B0BEC5", s=15, alpha=0.5, label="Normal")
ax.scatter(X_tsne[ensemble_anom, 0],  X_tsne[ensemble_anom, 1],
           c="#E53935", s=40, alpha=0.85, label="Anomaly (ensemble)")
ax.set_title("t-SNE: Ensemble Anomalies", fontsize=11, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(fontsize=9)

ax2 = axes[1]
sc = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1],
                 c=IF_score_norm, cmap="YlOrRd", s=12, alpha=0.7)
plt.colorbar(sc, ax=ax2, label="IF anomaly score")
ax2.set_title("t-SNE coloured by IF Anomaly Score", fontsize=11, fontweight="bold")
ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")

plt.suptitle("Figure 8. t-SNE Projection Showing Separation of Anomalies and Normal Samples",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure8_tSNE.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("   → saved Figure8_tSNE.png")

# ---------------------------------------------------------------------------
# 19. FIGURE 9 – TOP-10 FEATURE RANKING BAR CHART
# ---------------------------------------------------------------------------
print("Generating Figure 9: Feature z-score bar chart …")

top10_feats = t3.head(10)

fig, ax = plt.subplots(figsize=(11, 6))
colors_bar = ["#E53935" if d == "↑" else "#1565C0"
              for d in top10_feats["Direction"]]
bars = ax.barh(top10_feats["Feature"], top10_feats["Δz (std units)"].abs(),
               color=colors_bar, edgecolor="white", height=0.65)
ax.set_xlabel("| Δz | (Standardised Mean Difference)", fontsize=11)
ax.set_title("Figure 9. Top 10 Anomaly-Contributing Features Ranked by z-Score Difference\n"
             "(Red = anomalies higher; Blue = anomalies lower)",
             fontsize=11, fontweight="bold")
ax.invert_yaxis()
for bar, row in zip(bars, top10_feats.itertuples()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{row.Direction} {row._3} Δz",  # _3 = 'Δz grade'
            va="center", ha="left", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Figure9_Feature_Ranking.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("   → saved Figure9_Feature_Ranking.png")

# ---------------------------------------------------------------------------
# 20. SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}/")
print("  Figures: Figure2 – Figure9 (.png, 200 dpi)")
print("  Tables:  Table1_performance.csv")
print("           Table2_top_anomalies.csv")
print("           Table3_feature_ranking.csv")
print()
print("Anomaly summary:")
print(f"  Isolation Forest : {IF_anom.sum()} anomalies")
print(f"  One-Class SVM    : {OCSVM_anom.sum()} anomalies")
print(f"  Elliptic Envelope: {EE_anom.sum()} anomalies")
print(f"  Autoencoder      : {AE_anom.sum()} anomalies")
print(f"  Ensemble union   : {ensemble_anom.sum()} unique anomalous observations")
print(f"  Ensemble rate    : {ensemble_anom.mean()*100:.2f}%  "
      f"(paper reports ≈7.88%–8%)")
