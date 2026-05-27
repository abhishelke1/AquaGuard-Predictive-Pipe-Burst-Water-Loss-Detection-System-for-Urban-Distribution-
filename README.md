hello
# AquaGuard
updated
Predictive Pipe Burst and Water Loss Detection System for Urban Water Distribution Networks.

AquaGuard is an end-to-end machine learning project that simulates municipal water pipeline data, trains multiple predictive models, explains predictions with SHAP, and serves live risk scoring through a Flask web app and REST API.

## Highlights

- Synthetic infrastructure + sensor dataset generation (10,000 pipe segments)
- Data quality handling: missing-value imputation + outlier capping
- Imbalance mitigation with SMOTE
- Multi-model benchmarking:
	- Logistic Regression
	- Random Forest
	- Gradient Boosting
	- XGBoost (best)
- Explainability with SHAP (global and local)
- Time-series leak early warning with ARIMA pressure forecasting
- Production-style Flask dashboard + prediction API

## Project Structure

```text
AquaGuard/
|- app.py
|- aquaguard_main.py
|- aquaguard_dataset.csv
|- model_comparison.csv
|- requirements.txt
|- README.md
|- models/
|  |- model_metadata.json
|  |- pca.pkl
|  |- scaler.pkl
|  |- xgboost_aquaguard.pkl
|- outputs/
|  |- aquaguard_dataset.csv
|  |- model_comparison.csv
|- plots/
	 |- arima_forecast.png
	 |- correlation_heatmap.png
	 |- eda_distributions.png
	 |- model_evaluation.png
	 |- pca_scree.png
	 |- shap_bar.png
	 |- shap_summary.png
	 |- shap_waterfall.png
```

## Model Performance

Best model: XGBoost

- Test ROC-AUC: 0.9858
- Test F1: 0.8042
- Test Average Precision: 0.9091
- Top SHAP Features:
	- pipe_age_yrs
	- pressure_variance
	- material_enc

Model comparison snapshot:

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 | Test Avg-P |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9737 ± 0.0035 | 0.9687 | 0.5735 | 0.7499 |
| Random Forest | 0.9994 ± 0.0002 | 0.9786 | 0.7109 | 0.8253 |
| Gradient Boosting | 0.9994 ± 0.0002 | 0.9836 | 0.8031 | 0.8903 |
| XGBoost | 0.9997 ± 0.0001 | 0.9858 | 0.8042 | 0.9091 |

## Pipeline Overview

1. Generate synthetic water network data with realistic failure logic.
2. Run EDA and export diagnostic plots.
3. Preprocess:
	 - Median imputation for sensor dropouts
	 - IQR clipping for selected outliers
	 - Label encoding for categorical fields
4. Split, scale, and rebalance training data using SMOTE.
5. Train and compare multiple classifiers with cross-validation.
6. Evaluate on holdout test data (ROC, PR, confusion matrix).
7. Compute SHAP explanations.
8. Forecast pressure trend using ARIMA for leak warning signals.
9. Save trained model artifacts for inference.

## Quick Start

### 1) Clone

```bash
git clone https://github.com/abhishelke1/AquaGuard-Predictive-Pipe-Burst-Water-Loss-Detection-System-for-Urban-Distribution-.git
cd AquaGuard-Predictive-Pipe-Burst-Water-Loss-Detection-System-for-Urban-Distribution-
```

### 2) Create Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3) Install Dependencies

```bash
pip install -r requirements.txt
```

### 4) Run Training + Artifact Generation

```bash
python aquaguard_main.py
```

This generates:
- model files in models/
- charts in plots/
- CSV outputs in outputs/

### 5) Start Web App

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

## REST API

Base URL: http://127.0.0.1:5000

### Health Check

```http
GET /health
```

### Model Info

```http
GET /model_info
```

### Predict Failure Risk

```http
POST /predict
Content-Type: application/json
```

Sample request body:

```json
{
	"pipe_age_yrs": 45,
	"diameter_mm": 100,
	"burial_depth_m": 2.1,
	"num_prior_repairs": 3,
	"avg_pressure_bar": 3.2,
	"pressure_variance": 1.8,
	"avg_flow_lps": 25,
	"flow_variance": 5.1,
	"pressure_drop_pct": 12.5,
	"soil_temp_c": 20,
	"elevation_m": 150,
	"material": "cast_iron",
	"soil_type": "clay",
	"zone": "Zone_A"
}
```

## Output Artifacts

- Dataset outputs:
	- outputs/aquaguard_dataset.csv
	- outputs/model_comparison.csv
- Model artifacts:
	- models/xgboost_aquaguard.pkl
	- models/scaler.pkl
	- models/pca.pkl
	- models/model_metadata.json
- Plots:
	- plots/eda_distributions.png
	- plots/correlation_heatmap.png
	- plots/pca_scree.png
	- plots/model_evaluation.png
	- plots/arima_forecast.png
	- plots/shap_summary.png
	- plots/shap_bar.png
	- plots/shap_waterfall.png

## Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn, Imbalanced-learn
- XGBoost
- SHAP
- Statsmodels (ARIMA)
- Matplotlib, Seaborn
- Flask

## Use Cases

- Utility operations risk triage
- Proactive maintenance planning
- Leak early-warning intelligence
- Explainable AI support for field teams

## Roadmap

- Integrate real SCADA/IoT streams
- Add geospatial visualization (GIS)
- Add model drift monitoring
- Containerize deployment with Docker
- Add CI validation for model training and API

## License

This project is intended for educational and research use. Add a license file if you plan public reuse in production settings.
