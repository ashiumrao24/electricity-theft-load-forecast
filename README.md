# ⚡ Electricity Load Forecasting & Theft Detection

A compact end-to-end project demonstrating **time-series forecasting**, **electricity theft anomaly detection**, and an interactive **Streamlit dashboard**.  
Built using the UCI Household Power Consumption dataset.

---

## 🌟 Key Features
- Hourly load forecasting using **XGBoost**
- Synthetic electricity theft generation
- Theft detection classifier using residual features
- Clean feature engineering pipeline
- Interactive Streamlit dashboard
- Tableau dasboard

---

## 📁 Project Structure
```markdown
📦 electricity-theft-load-forecast
│
├── 📂 app/                 — Streamlit dashboard
│   └── app.py
│
├── 📂 src/                 — Preprocessing + ML models
│   ├── preprocess.py
│   ├── models_forecast.py
│   └── models_theft.py
│
├── 📄 requirements.txt     — Python dependencies
├── 📄 .gitignore           — Ignores dataset, models, venv
└── 📄 README.md            — Project documentation
```

---
## 📊 Dataset

This project uses the **UCI Household Electric Power Consumption Dataset**, a large real-world dataset containing over **2 million minute-level energy readings** from a single household over 4 years.

### 📌 Dataset Source  
UCI Machine Learning Repository / Kaggle Mirror

📥 **Download Dataset:**  
https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

---

### 📂 How to Add the Dataset

After downloading, place the raw file here in your project:
```bash
data/household_power_consumption.txt
```

⚠️ **Important:**  
- This dataset is **not included** in the GitHub repository because it is large.  
- It is ignored using `.gitignore` to keep the repository clean.

---

### 📑 Dataset Details
The dataset includes:

| Feature | Description |
|---------|-------------|
| Date | Date of measurement |
| Time | Time of measurement |
| Global_active_power | Household global minute-averaged active power (kW) |
| Global_reactive_power | Reactive power (kW) |
| Voltage | Voltage (V) |
| Global_intensity | Current intensity (A) |
| Sub_metering_1 | Energy sub-metering No.1 (Wh) |
| Sub_metering_2 | Energy sub-metering No.2 (Wh) |
| Sub_metering_3 | Energy sub-metering No.3 (Wh) |

The preprocessing script converts this minute-level data into **hourly kWh** with additional engineered features for forecasting and theft detection.

---
## 🔧 Workflow Overview

## 🧹 1️⃣ Data Preprocessing (`src/preprocess.py`)

### Steps performed:
- Load raw dataset (`household_power_consumption.txt`)
- Combine Date + Time → unified datetime column
- Handle missing or invalid values
- Convert raw minute-level power to **hourly kWh**
  - `energy_kwh = (Global_active_power * 1/60)`
- Generate engineered ML features:
  - Hour of day
  - Day of week
  - Month
  - Weekend flag
  - Lag features:  
    - `lag_1h`, `lag_24h`, `lag_168h`
  - Rolling statistics:
    - 24-hour rolling mean
    - 24-hour rolling standard deviation

### Output:
```bash
data/hourly_energy_features.csv
```

---

## 📈 2️⃣ Load Forecasting Model (`src/models_forecast.py`)

### What happens:
- Split processed hourly features into train/test sets
- Train **XGBoost Regressor** to predict hourly energy usage
- Evaluate model using:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
- Save trained model

### Output:
```bash
models/xgb_forecast.pkl
```

### Example Performance:
| Metric | Value |
|--------|--------|
| MAE | ~0.32 |
| RMSE | ~0.46 |

---

## ⚠️ 3️⃣ Electricity Theft Simulation & Detection (`src/models_theft.py`)

### Theft Simulation:
- Randomly pick ~5% of timestamps
- Reduce consumption by a random factor (20%–70%)
- Create synthetic labels:
  - `is_theft = 1` for tampered readings
  - `is_theft = 0` otherwise

### Feature Engineering for Theft Detection:
- Forecasted consumption (from model)
- Residual = Actual − Forecast
- Absolute residual
- Residual ratio
- Negative residual flag
- Rolling residual features (3h, 24h)
- Time-based features

### Model:
- **XGBoost Classifier**  
- Handles imbalance using `scale_pos_weight`

### Outputs:
```bash
models/xgb_theft.pkl
data/final_powerbi_dataset.csv
```

This exported dataset is used for Power BI reporting.

---

## 🌐 4️⃣ Streamlit Dashboard (`app/app.py`)

### Dashboard provides:
- Actual vs Forecast charts
- Suspicious hour detection with probability thresholds
- Detailed 48-hour contextual view
- Interactive selection of timestamps
- Data exploration tools for anomalies

Run the app:
```bash
streamlit run app/app.py
```
---

## 📊 5️⃣ Interactive Tableau Dashboard

🔗 **Live Dashboard (Tableau Public):**  
https://public.tableau.com/app/profile/aashi.umrao/viz/Electricity_Theft_Detection_Dashboard/Dashboard1

### 🖼️ Dashboard Preview
![Dashboard Preview](assets/dashboard_preview.jpg)

> Use the Date Range and Theft Risk Threshold filters to explore high-risk electricity
> consumption patterns and identify suspicious hours.










