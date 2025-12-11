import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_PATH = Path("data/household_power_consumption.txt")  # your file
OUT_PATH = Path("data/hourly_energy_features.csv")

def load_raw_data():
    print("-> Loading raw data from", DATA_PATH, file=sys.stderr)
    df = pd.read_csv(
        DATA_PATH,
        sep=';',
        na_values='?',
        low_memory=False
    )
    print("   raw shape:", df.shape, file=sys.stderr)
    df['datetime'] = pd.to_datetime(
        df['Date'] + " " + df['Time'],
        format="%d/%m/%Y %H:%M:%S",
        errors='coerce'
    )
    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')
    df = df.sort_index()

    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols, how='all')
    print("   after cleaning shape:", df.shape, file=sys.stderr)
    return df

def resample_hourly(df):
    print("-> Resampling to hourly energy (kWh)...", file=sys.stderr)
    power = df["Global_active_power"]
    # convert kW-minute -> kWh (kW * 1/60 hour)
    energy_kwh = (power * (1.0/60.0)).resample("H").sum()
    hourly = pd.DataFrame({"energy_kwh": energy_kwh})
    hourly = hourly.dropna()
    print("   hourly shape:", hourly.shape, file=sys.stderr)
    return hourly

def add_features(df):
    print("-> Adding time features and lags...", file=sys.stderr)
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df['dayofweek'] >= 5).astype(int)

    df["lag_1"] = df["energy_kwh"].shift(1)
    df["lag_24"] = df["energy_kwh"].shift(24)
    df["lag_168"] = df["energy_kwh"].shift(168)

    df["rollmean_24"] = df["energy_kwh"].rolling(24).mean()
    df["rollstd_24"] = df["energy_kwh"].rolling(24).std()

    df = df.dropna()
    print("   features shape (after dropna):", df.shape, file=sys.stderr)
    return df

def prepare_dataset():
    raw = load_raw_data()
    hourly = resample_hourly(raw)
    final = add_features(hourly)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PATH)
    print(f"Saved {OUT_PATH} (shape: {final.shape})", file=sys.stderr)

if __name__ == "__main__":
    prepare_dataset()
