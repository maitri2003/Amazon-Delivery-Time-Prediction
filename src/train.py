"""
Model Training Script for Amazon Delivery Time Prediction
---------------------------------------------------------
- Load processed data
- Split features and target
- Train model using RandomForestRegressor
- Evaluate performance
- Save trained model + feature names
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ============================================================
# Function: Split features & target
# ============================================================
def split_features_target(df, target_col="Delivery_Duration_min"):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset!")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ============================================================
# Function: Train Model
# ============================================================
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# Function: Evaluate Model
# ============================================================
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print("\n📊 Model Evaluation Results:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# Main Function
# ============================================================
def main():
    print("[INFO] Loading processed data...")
    input_path = "../data/processed/amazon_delivery_clean.csv"
    df = pd.read_csv(input_path)

    print("[INFO] Splitting features and target...")
    X, y = split_features_target(df)

    # ✅ Keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64"])

    # ✅ Print columns used for training
    print("\n[INFO] Feature columns used for training:")
    print(list(X.columns))

    print("[INFO] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Training model...")
    model = train_model(X_train, y_train)

    print("[INFO] Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("[INFO] Saving trained model and feature names...")
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/random_forest_model.pkl")
    joblib.dump(list(X.columns), "../models/feature_names.pkl")

    print("[INFO] Model saved at ../models/random_forest_model.pkl")
    print("[INFO] Feature names saved at ../models/feature_names.pkl")


if __name__ == "__main__":
    main()
