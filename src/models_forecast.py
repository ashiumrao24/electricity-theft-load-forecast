import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

DATA_PATH = Path("data/hourly_energy_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_forecast.pkl"

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")
    return df

def train_model():
    print("Loading data...")
    df = load_data()
    print("shape:", df.shape)

    target = "energy_kwh"
    features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    # time-based split (earliest 80% train, last 20% test)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print("Training set:", X_train.shape, "Test set:", X_test.shape)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        verbosity=1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved forecast model to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
