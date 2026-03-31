# National-Scale Anomaly Detection in Inland Waters
## An Explainable Multi-Model AI Framework for Environmental Governance

**Paper:** Bhowmik P.N., Varshney D., Mishra L., Anand P., Aggarwal M. (2025).  
*National-Scale Anomaly Detection in Inland Waters: An Explainable Multi-Model AI Framework for Environmental Governance.*

---

## Repository Contents

| File | Description |
|------|-------------|
| `water_quality_anomaly_detection.py` | **Main script** — reproduces all figures and tables |
| `Dataset.xlsx` | CPCB 2021 inland water-quality dataset (620 records, 32 states) |
| `requirements.txt` | Python package dependencies |
| `figures/` | Output folder — all figures (PNG) and tables (CSV) saved here |

---

## Dataset

The dataset is from the **Central Pollution Control Board (CPCB)** National Water Quality Monitoring Programme (NWMP), 2021. It covers inland lentic water bodies (lakes, tanks, ponds, wetlands) across 32 Indian states with the following parameters:

- Temperature (Min/Max), Dissolved Oxygen (Min/Max), pH (Min/Max)
- Conductivity (Min/Max), BOD (Min/Max), Nitrate+Nitrite (Min/Max)
- Fecal Coliform (Min/Max), Total Coliform (Min/Max)

**Source:** https://cpcb.nic.in

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python water_quality_anomaly_detection.py
```

All outputs are saved to `figures/`:

| Output | Description |
|--------|-------------|
| `Figure2_KDE_plots.png` | KDE distributions of normalised features |
| `Figure3_Correlation_Heatmap.png` | Pearson correlation matrix |
| `Figure4_Confusion_Matrices.png` | Confusion matrices for 4 models |
| `Figure5_Score_Histograms.png` | Normalised anomaly score histograms |
| `Figure6_PCA_Anomalies.png` | PCA projection of top anomalies |
| `Figure7_SHAP_Summary.png` | SHAP feature importance (Isolation Forest) |
| `Figure8_tSNE.png` | t-SNE cluster separation |
| `Figure9_Feature_Ranking.png` | Top-10 feature z-score ranking |
| `Table1_performance.csv` | Precision / Recall / F1 for all models |
| `Table2_top_anomalies.csv` | Top-10 anomalies ranked per model |
| `Table3_feature_ranking.csv` | Feature Δz and Δμ ranking |

---

## Models & Hyperparameters

| Model | Key Parameters |
|-------|---------------|
| Isolation Forest | `contamination=0.10`, `n_estimators=100`, `random_state=42` |
| One-Class SVM | `nu=0.10`, `kernel='rbf'`, `gamma=0.10` |
| Elliptic Envelope | `contamination=0.10`, `support_fraction=0.90`, `random_state=42` |
| Autoencoder | hidden=10 neurons, ReLU, Adam optimiser, 50 epochs, batch=32 |

All models use `contamination=0.10`, reflecting the expected ~10% anomaly rate in environmental monitoring datasets.

---

## Feature Engineering

Three domain-informed ecological indicators are constructed:

1. **BOD/DO ratio** = BOD_max / (DO_min + ε) — oxidative stress indicator
2. **DO deficit** = DO_sat − DO_min — oxygen stress indicator  
3. **Coliform Load Index (CLI)** = (Fecal Coliform_min + Total Coliform_min) / 2 — microbiological risk

---

## Reproducibility

All random seeds are fixed (`SEED=42`) for full reproducibility:
```python
np.random.seed(42)
tf.random.set_seed(42)
```

---

## License

MIT License — see LICENSE file.

---

## Citation

If you use this code or dataset, please cite:

```
Bhowmik P.N., Varshney D., Mishra L., Anand P., Aggarwal M. (2025).
National-Scale Anomaly Detection in Inland Waters: An Explainable Multi-Model 
AI Framework for Environmental Governance.
doi: [journal DOI]
```

Code archive: https://doi.org/10.5281/zenodo.[YOUR-ZENODO-ID]
