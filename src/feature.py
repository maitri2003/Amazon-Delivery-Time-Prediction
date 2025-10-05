# features.py
"""
Feature Engineering Script for Amazon Delivery Time Prediction
---------------------------------------------------------------
- Load raw dataset
- Create datetime features
- Create target (Delivery Duration in minutes)
- Add geospatial distance
- Encode categorical variables
- Save processed dataset
"""

import os
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Function: Haversine Distance (Store <-> Drop location)
# ============================================================
def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points (in km)."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in km
    return c * r


# ============================================================
# Function: Create Features
# ============================================================
def create_features(df):
    # ✅ Combine Order_Date + Order_Time
    if "Order_Date" in df.columns and "Order_Time" in df.columns:
        df["Order_Datetime"] = pd.to_datetime(
            df["Order_Date"].astype(str) + " " + df["Order_Time"].astype(str),
            errors="coerce"
        )

    # ✅ Convert Pickup_Time
    if "Pickup_Time" in df.columns:
        df["Pickup_Datetime"] = pd.to_datetime(df["Pickup_Time"], errors="coerce")

    # ✅ Target Variable (Delivery Duration in minutes)
    if "Delivery_Time" in df.columns:
        df["Delivery_Duration_min"] = df["Delivery_Time"] * 60
    elif "Pickup_Datetime" in df.columns:
        df["Delivery_Duration_min"] = (df["Pickup_Datetime"] - df["Order_Datetime"]).dt.total_seconds() / 60.0
    else:
        raise KeyError("Neither Delivery_Time nor Pickup_Datetime found to compute target")

    # ✅ Extract time-based features
    df["Order_Hour"] = df["Order_Datetime"].dt.hour
    df["Order_Weekday"] = df["Order_Datetime"].dt.dayofweek  # numeric weekday

    if "Pickup_Datetime" in df.columns:
        df["Pickup_Hour"] = df["Pickup_Datetime"].dt.hour

    # ✅ Geospatial Distance
    if {"Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude"}.issubset(df.columns):
        df["Distance_km"] = df.apply(
            lambda row: haversine(
                row["Store_Longitude"], row["Store_Latitude"],
                row["Drop_Longitude"], row["Drop_Latitude"]
            ),
            axis=1
        )

    # ======================================================
    # Encode Categorical Columns
    # ======================================================
    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]

    le = LabelEncoder()
    for col in categorical_cols:
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f"[WARN] Skipping {col} due to: {e}")

    return df


# ============================================================
# Main Function
# ============================================================
def main():
    print("[INFO] Loading raw data...")
    input_path = "../data/raw/amazon_delivery.csv"
    df = pd.read_csv(input_path)

    print("[INFO] Creating features...")
    df = create_features(df)

    print("[INFO] Saving processed data...")
    os.makedirs("../data/processed", exist_ok=True)
    output_path = "../data/processed/amazon_delivery_clean.csv"
    df.to_csv(output_path, index=False)

    print(f"[INFO] Processed dataset saved at: {output_path}")


if __name__ == "__main__":
    main()
