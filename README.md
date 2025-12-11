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
