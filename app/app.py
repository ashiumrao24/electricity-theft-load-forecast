import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Load Forecast & Theft Detection")

DATA_CSV = Path("data/final_powerbi_dataset.csv")
FORECAST_MODEL = Path("models/xgb_forecast.pkl")
THEFT_MODEL = Path("models/xgb_theft.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    return df

@st.cache_resource
def load_models():
    f_model = joblib.load(FORECAST_MODEL)
    t_model = joblib.load(THEFT_MODEL)
    return f_model, t_model

def main():
    st.title("Electricity Load Forecasting & Theft Detection")
    st.markdown("**Models:** XGBoost forecast & XGBoost theft detector. Data: UCI household dataset (hourly).")

    if not DATA_CSV.exists():
        st.error(f"Data not found: {DATA_CSV}. Run the preprocessing & model scripts first.")
        return

    df = load_data()
    f_model, t_model = load_models()

    col1, col2 = st.columns([3,1])
    with col2:
        st.subheader("Controls")
        min_date = df.index.date.min()
        max_date = df.index.date.max()
        start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)
        thr = st.slider("Theft probability threshold", 0.0, 1.0, 0.7, 0.01)

    # filter
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    df_range = df.loc[mask].copy()
    if df_range.empty:
        st.warning("No data in selected date range.")
        return

    # Plot actual vs forecast
    with col1:
        st.subheader("Actual vs Forecast")
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_range.index, df_range["energy_kwh"], label="Actual")
        if "forecast" in df_range.columns:
            ax.plot(df_range.index, df_range["forecast"], linestyle="--", label="Forecast")
        ax.set_ylabel("Energy (kWh)")
        ax.legend()
        st.pyplot(fig)

    # Suspicious points
    st.subheader("Suspicious Hours (by theft_proba)")
    df_range = df_range.assign(theft_proba = df_range.get("theft_proba", 0.0))
    suspicious = df_range[df_range["theft_proba"] >= thr].sort_values("theft_proba", ascending=False)

    st.write(f"Suspicious hours found: {len(suspicious)} (threshold = {thr:.2f})")
    if not suspicious.empty:
        st.dataframe(suspicious[["energy_kwh","forecast","residual","abs_residual","theft_proba","is_theft"]].head(200))

        row = st.selectbox("Pick one suspicious timestamp for details", suspicious.index[:200].to_list())
        if row is not None:
            sample = df.loc[row:row]  # single-row DataFrame
            st.markdown("**Selected timestamp details**")
            st.write(sample.T)

            # show small plot around the sample (±24h)
            window = df.loc[row - pd.Timedelta(hours=24) : row + pd.Timedelta(hours=24)]
            fig2, ax2 = plt.subplots(figsize=(10,3))
            ax2.plot(window.index, window["energy_kwh"], label="Actual")
            if "forecast" in window.columns:
                ax2.plot(window.index, window["forecast"], linestyle="--", label="Forecast")
            ax2.axvline(row, color="red", linestyle=":", label="Selected")
            ax2.legend()
            st.pyplot(fig2)
    else:
        st.info("No suspicious hours above threshold in selected range.")

    st.markdown("---")
    st.caption("Tip: adjust the theft probability threshold to trade-off false positives vs false negatives.")

if __name__ == "__main__":
    main()
