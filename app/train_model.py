import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("data/raw_papers.csv")

# Rename columns to match the application
column_map = {
        "No.": "publication_id",
        "Material": "material_name",
        "Density Measurement Method": "density_method",
        "Printed Geom.": "printed_geometry",
        "Laser Power (W)": "laser_power_w",
        "Scan Speed (mm/s)": "scan_speed_mm_s",
        "Hatch space (mm)": "hatch_distance_mm",
        "Layer thickness (mm)": "layer_thickness_mm",
        "Spot size (mm)": "spot_size_mm",
        "Geometric Factor": "geometric_factor",
        "D50 μm": "d50_um",
        "Porosity(%)": "porosity_pct",
        "RD (%)": "relative_density_pct",
        "Printer Model": "printer_model",
        "DOI": "doi",
        "Energy density (J/mm³)": "ved_j_mm3",
        "Microhadness (HV)": "microhardness_hv",
        # "Hatch Angle(degree)": "hatch_angle_deg",
        # "Exposure time in µs": "exposure_time_us",
    }

df = df.rename(columns=column_map)

print("\nOriginal porosity values:")
print(df["porosity_pct"].dropna().head(30).to_list())

print("\nUnique porosity formats:")
print(df["porosity_pct"].dropna().astype(str).unique()[:30])

# Features we want to investigate
features = [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
    "spot_size_mm",
]

target = "porosity_pct"

# Extract leading numeric porosity value without parsing std
porosity_values = df["porosity_pct"].astype(str)
df["porosity_pct"] = pd.to_numeric(
    porosity_values.str.extract(r"^\s*([+-]?\d+(?:\.\d+)?)")[0],
    errors="coerce",
)

# Convert features to numeric
for column in features:
  df[column] = pd.to_numeric(df[column], errors="coerce")


# Analyse data completeness by material
analysis_columns = [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
    "spot_size_mm",
    "porosity_pct",
]

print("\n===== DATA COMPLETENESS BY MATERIAL =====")

for material in sorted(df["material_name"].dropna().unique()):
  material_df = df[df["material_name"] == material]

  print(f"\nMaterial: {material}")
  print(f"Total records: {len(material_df)}")

  for column in analysis_columns:
    available = material_df[column].notna().sum()
    percentage = available / len(material_df) * 100

    print(
        f"{column}: {available}/{len(material_df)} "
        f"({percentage:.1f}% available)"
    )


# ===== MODEL TRAINING & EXPORT =====
train_df = df[features + [target]].dropna(subset=[target]).copy()

# Fill missing feature values using training medians so sparse columns do not drop rows
feature_medians = {}
for col in features:
  median_val = train_df[col].median()
  feature_medians[col] = float(median_val) if pd.notna(median_val) else 0.0
  train_df[col] = train_df[col].fillna(feature_medians[col])

print(f"\nRows available for training: {len(train_df)}")

if len(train_df) >= 5:
  X = train_df[features]
  y = train_df[target]

  if len(train_df) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Test MAE: {mae:.4f}")
    if len(y_test) >= 2:
      print(f"Test R2:  {r2_score(y_test, preds):.4f}")
  else:
    # Fit on all labeled rows when sample count is very small
    model = XGBRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X, y)
    print("Small sample size (<10): fitted model directly on all labeled rows.")

  os.makedirs("models", exist_ok=True)
  model_path = os.path.join("models", "porosity_xgb_model.pkl")
  artifact = {
      "model": model,
      "features": features,
      "target": target,
      "feature_medians": feature_medians,
  }
  joblib.dump(artifact, model_path)
  print(f"Saved model artifact to: {model_path}")
else:
  print(
      "Not enough valid porosity rows (<5) to train a regression model yet."
  )