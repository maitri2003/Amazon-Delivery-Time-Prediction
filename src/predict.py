# predict.py
"""
Prediction Script for Amazon Delivery Time Prediction
-----------------------------------------------------
- Load trained model
- Load new input data
- Apply same preprocessing
- Predict delivery time in minutes
"""

import pandas as pd
import joblib
import os


# ============================================================
# Function: Load trained model
# ============================================================
def load_model(model_path="../models/random_forest_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    print("[INFO] Loading trained model...")
    model = joblib.load(model_path)
    return model


# ============================================================
# Function: Preprocess new data (same as feature.py logic)
# ============================================================
def preprocess_input_data(df):
    print("[INFO] Preprocessing input data...")

    # Convert datetimes
    df["Order_Datetime"] = pd.to_datetime(df["Order_Datetime"], errors="coerce")
    df["Pickup_Datetime"] = pd.to_datetime(df["Pickup_Datetime"], errors="coerce")

    # Extract datetime features
    df["Order_Hour"] = df["Order_Datetime"].dt.hour
    df["Order_Weekday"] = df["Order_Datetime"].dt.dayofweek
    df["Pickup_Hour"] = df["Pickup_Datetime"].dt.hour
    df["Pickup_Weekday"] = df["Pickup_Datetime"].dt.dayofweek

    # Compute difference
    df["Time_to_Pickup_min"] = (
        (df["Pickup_Datetime"] - df["Order_Datetime"]).dt.total_seconds() / 60
    )

    # Drop unnecessary columns
    drop_cols = ["Order_ID", "Order_Datetime", "Pickup_Datetime"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # Keep only numeric features
    df = df.select_dtypes(include=["int64", "float64"])

    return df


# ============================================================
# Function: Make predictions
# ============================================================
def predict_delivery_time(model, df):
    print("[INFO] Making predictions...")
    predictions = model.predict(df)
    df["Predicted_Delivery_Duration_min"] = predictions
    return df


# ============================================================
# Main function
# ============================================================
def main():
    # Example: new unseen input data (you can replace with your own CSV)
    input_path = "../data/processed/amazon_delivery_clean.csv"

    if not os.path.exists(input_path):
        print(f"[WARNING] No new_orders.csv found! Creating sample input...")
        data = {
            "Order_ID": ["A123", "A124"],
            "Order_Datetime": ["2022-03-14 10:30:00", "2022-03-15 18:00:00"],
            "Pickup_Datetime": ["2022-03-14 11:00:00", "2022-03-15 18:25:00"],
            "Distance_km": [12.5, 8.2],
            "Weather": [1, 0],  # Example: 1 = rainy, 0 = clear
            "Traffic": [2, 1],  # Example: 2 = heavy, 1 = moderate
        }
        df = pd.DataFrame(data)
    else:
        print("[INFO] Loading new input data...")
        df = pd.read_csv(input_path)

    # Load model
    model = load_model()

    # Preprocess input
    df_processed = preprocess_input_data(df)

    # Predict
    result = predict_delivery_time(model, df_processed)

    # Save results
    os.makedirs("../data/predictions", exist_ok=True)
    output_path = "../data/predictions/predicted_delivery_times.csv"
    result.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved at: {output_path}\n")

    print(result[["Predicted_Delivery_Duration_min"]])


if __name__ == "__main__":
    main()
