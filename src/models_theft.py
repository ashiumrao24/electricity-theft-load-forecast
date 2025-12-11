import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from xgboost import XGBClassifier

# Paths
DATA_PATH = Path("data/hourly_energy_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
FORECAST_MODEL_PATH = MODEL_DIR / "xgb_forecast.pkl"
THEFT_MODEL_PATH = MODEL_DIR / "xgb_theft.pkl"
FINAL_EXPORT = Path("data/final_powerbi_dataset.csv")

# Synthetic theft generation settings (tweak if you like)
THEFT_FRACTION = 0.05   # fraction of hours to tamper
MIN_SCALE = 0.2         # tamper multiplier min (reduce to 20% of real)
MAX_SCALE = 0.7         # tamper multiplier max (reduce to 70% of real)
RANDOM_STATE = 42

def load_features():
    print("Loading features from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")
    print("shape:", df.shape)
    return df

def simulate_theft(df, theft_fraction=THEFT_FRACTION, min_scale=MIN_SCALE, max_scale=MAX_SCALE, random_state=RANDOM_STATE):
    print(f"Simulating theft: fraction={theft_fraction}, scale_range=[{min_scale},{max_scale}]")
    np.random.seed(random_state)
    df = df.copy()
    n = len(df)
    n_theft = int(n * theft_fraction)
    theft_indices = np.random.choice(df.index, size=n_theft, replace=False)

    # initialize label column
    df["is_theft"] = 0

    # scales for each tampered index
    scales = np.random.uniform(min_scale, max_scale, size=n_theft)
    for idx, scale in zip(theft_indices, scales):
        # reduce recorded energy to simulate theft (keeping other features intact)
        df.at[idx, "energy_kwh"] = df.at[idx, "energy_kwh"] * scale
        df.at[idx, "is_theft"] = 1

    print("  simulated theft count:", df["is_theft"].sum())
    return df

def add_forecast_features(df):
    print("Loading forecast model:", FORECAST_MODEL_PATH)
    forecast_model = joblib.load(FORECAST_MODEL_PATH)

    # features used in forecast model: all columns except target 'energy_kwh'
    target = "energy_kwh"
    feature_cols = [c for c in df.columns if c != target and c != "is_theft"]

    # ensure dtype alignment
    X_feat = df[feature_cols].copy()
    # if any NaNs present, fill (should not normally be present)
    X_feat = X_feat.fillna(method="ffill").fillna(0)

    print("Computing forecast predictions for residuals...")
    preds = forecast_model.predict(X_feat)
    df["forecast"] = preds
    df["residual"] = df["energy_kwh"] - df["forecast"]
    df["abs_residual"] = df["residual"].abs()

    # extra engineered features for theft detection
    df["residual_ratio"] = df["residual"] / (df["forecast"].replace(0, np.nan)).fillna(0)
    df["is_negative_residual"] = (df["residual"] < 0).astype(int)
    # rolling residual behavior
    df["rollresid_3h"] = df["residual"].rolling(3, min_periods=1).mean()
    df["rollresid_24h"] = df["residual"].rolling(24, min_periods=1).mean()

    print("Added forecast-based features.")
    return df

def train_theft_classifier(df):
    # prepare features (drop columns we shouldn't use)
    drop_cols = ["is_theft", "forecast"]  # target is is_theft
    target = "is_theft"
    feature_cols = [c for c in df.columns if c not in drop_cols and c != target]

    X = df[feature_cols].fillna(0)
    y = df[target].astype(int)

    print("Feature matrix shape:", X.shape, "Positive labels:", int(y.sum()))

    # time-independent split but preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # handle class imbalance using scale_pos_weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = max(1.0, neg / max(1, pos))

    print("Training theft classifier... scale_pos_weight:", scale_pos_weight)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nClassification report on test set:")
    print(classification_report(y_test, y_pred, digits=4))

    # precision-recall AUC
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # confusion matrix
    print("Confusion matrix (tn, fp, fn, tp):")
    print(confusion_matrix(y_test, y_pred).ravel())

    # save model
    joblib.dump(clf, THEFT_MODEL_PATH)
    print("Saved theft model to:", THEFT_MODEL_PATH)

    # attach predictions back to dataframe for export
    return clf, X_test.index, y_test, y_proba

def export_for_powerbi(df):
    print("Exporting final Power BI dataset to:", FINAL_EXPORT)
    # select and export useful columns
    export_cols = [
        "energy_kwh", "forecast", "residual", "abs_residual",
        "residual_ratio", "is_negative_residual", "rollresid_3h",
        "rollresid_24h", "is_theft"
    ]
    df_export = df.copy()
    # ensure export includes datetime as a column
    df_export.reset_index(inplace=True)
    df_export.to_csv(FINAL_EXPORT, index=False)
    print("Export saved. shape:", df_export.shape)

def main():
    df = load_features()
    df = simulate_theft(df)
    df = add_forecast_features(df)
    clf, test_idx, y_test, y_proba = train_theft_classifier(df)
    # add probability column for entire df (use model on all)
    feature_cols_all = [c for c in df.columns if c not in ["is_theft", "forecast"]]
    df_features_all = df[feature_cols_all].fillna(0)
    df["theft_proba"] = clf.predict_proba(df_features_all)[:, 1]
    export_for_powerbi(df)

if __name__ == "__main__":
    main()
