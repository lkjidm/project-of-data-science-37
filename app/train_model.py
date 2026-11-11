import os
import joblib
import pandas as pd
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("data/raw_papers.csv")

# Rename columns to match the application
column_map = {
    "No.": "publication_id",
    "Material": "material_name",
    "Laser Power (W)": "laser_power_w",
    "Scan Speed (mm/s)": "scan_speed_mm_s",
    "Hatch space (mm)": "hatch_distance_mm",
    "Layer thickness (mm)": "layer_thickness_mm",
    "Energy density (J/mm³)": "ved_j_mm3",
    "Porosity in vol%": "porosity_pct",
    "Microhadness (HV)": "microhardness_hv",
    "Spot size (mm)": "spot_size_mm",
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
    "ved_j_mm3",
    "spot_size_mm",
]

target = "porosity_pct"

# Extract mean porosity and standard deviation
porosity_values = df["porosity_pct"].astype(str)

df["porosity_std"] = pd.to_numeric(
    porosity_values.str.extract(r"±\s*([\d.]+)")[0],
    errors="coerce"
)

df["porosity_pct"] = pd.to_numeric(
    porosity_values.str.extract(r"^\s*([\d.]+)")[0],
    errors="coerce"
)

# Convert remaining features to numeric
for column in features:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Calculate missing VED values using process parameters
missing_ved = df["ved_j_mm3"].isna()

df.loc[missing_ved, "ved_j_mm3"] = (
    df.loc[missing_ved, "laser_power_w"]
    /
    (
        df.loc[missing_ved, "scan_speed_mm_s"]
        * df.loc[missing_ved, "hatch_distance_mm"]
        * df.loc[missing_ved, "layer_thickness_mm"]
    )
)

# Analyse data completeness by material
analysis_columns = [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
    "spot_size_mm",
    "ved_j_mm3",
    "porosity_pct",
    "porosity_std",
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

# Descriptive statistics by material
comparison_columns = [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
    "spot_size_mm",
    "ved_j_mm3",
]

print("\n===== MATERIAL COMPARISON =====")

for material, group in df.groupby("material_name"):
    print(f"\n{material}")
    print(
        group[comparison_columns]
        .agg(["mean", "median", "std", "min", "max"])
        .round(3)
    )
    
# Check data availability
print("\nUsable measurements per column:")
print(df[features + [target]].count())

print("\nUsable porosity measurements by material:")
print(df.groupby("material_name")["porosity_pct"].count())

print("\nRows with all required values:")
print(df[features + [target]].dropna().shape[0])

print("\nExtracted porosity:")
print(df[["porosity_pct", "porosity_std"]].dropna())

print("\nUsable porosity measurements:")
print(df["porosity_pct"].count())

print("\nFive records with porosity measurements:")
print(
    df.loc[
        df["porosity_pct"].notna(),
        [
            "material_name",
            "laser_power_w",
            "scan_speed_mm_s",
            "hatch_distance_mm",
            "layer_thickness_mm",
            "ved_j_mm3",
            "porosity_pct",
            "porosity_std",
        ],
    ].to_string(index=True)
)

print("\nRecovered VED values:")
print(
    df.loc[
        df["porosity_pct"].notna(),
        ["material_name", "ved_j_mm3", "porosity_pct"]
    ]
)