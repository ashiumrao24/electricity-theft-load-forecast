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
- Power BI dashboard (coming soon)

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




