import os
import joblib
import pandas as pd
from xgboost import XGBRegressor

# 1. Load your curated dataset
df = pd.read_csv("data/raw_papers.csv")

# 2. Define features (X) and target (y)
features = [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
    "ved_j_mm3",
]
target = "porosity_pct"

X = df[features]
y = df[target]

# 3. Train an XGBoost Regressor (well-suited for small datasets)
model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X, y)

# 4. Save the trained model to the models directory
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_porosity.pkl")

print("Model successfully trained and saved to models/xgb_porosity.pkl")