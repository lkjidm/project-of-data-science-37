from pathlib import Path
import re
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Page Configuration & Layout Setup
st.set_page_config(
    page_title="LPBF Research Portal & ML Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main Title
st.title("🔬 Metal LPBF Literature Database & ML Simulator")
st.markdown(
    "A research-focused MVP integrating curated literature data (316L & 18Ni300)"
    " with machine learning-driven process parameter prediction."
)


def _safe_median(series: pd.Series, fallback: float) -> float:
  # Safely compute median even if column is missing or all values are NaN
  numeric_col = pd.to_numeric(series, errors="coerce")
  med = numeric_col.median()
  return float(med) if pd.notna(med) else fallback


# 2. Data Loading Function with Caching
@st.cache_data
def load_literature_data():
  # Load the curated dataset relative to project root
  base_dir = Path(__file__).resolve().parent.parent
  csv_path = base_dir / "data" / "raw_papers.csv"

  if csv_path.exists():
    data = pd.read_csv(csv_path)

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
        "Microhadness (HV)": "microhardness_hv",
    }

    data = data.rename(columns=column_map)
    return data

  # Fallback mock data using 316L and 18Ni300 when CSV is unavailable
  return pd.DataFrame({
      "publication_id": [
          "Ref_316L_01",
          "Ref_18Ni_01",
          "Ref_316L_02",
          "Ref_18Ni_02",
          "Ref_316L_03",
          "Ref_18Ni_03",
      ],
      "material_name": [
          "316L",
          "18Ni300",
          "316L",
          "18Ni300",
          "316L",
          "18Ni300",
      ],
      "laser_power_w": [200.0, 230.0, 280.0, 300.0, 150.0, 180.0],
      "scan_speed_mm_s": [800.0, 1000.0, 1400.0, 1500.0, 600.0, 1300.0],
      "hatch_distance_mm": [0.10, 0.10, 0.11, 0.10, 0.09, 0.11],
      "layer_thickness_mm": [0.03, 0.03, 0.04, 0.03, 0.03, 0.04],
      "spot_size_mm": [0.08, 0.07, 0.08, 0.08, 0.07, 0.08],
      "d50_um": [30.0, 35.0, 32.0, 38.0, 28.0, 34.0],
      "porosity_pct": [0.12, 0.18, 0.85, 1.35, 0.35, 2.10],
      "microhardness_hv": [235.0, 360.0, 220.0, 345.0, 240.0, 330.0],
  })


df = load_literature_data()

# 3. Sidebar: Control Panel (Filters & Simulator)
st.sidebar.header("🎛️ Control Panel")

# Section A: Global Literature Filters
st.sidebar.subheader("1. Literature Filter")
available_materials = sorted(df["material_name"].dropna().astype(str).unique())
if not available_materials:
  available_materials = ["316L", "18Ni300"]

selected_material = st.sidebar.selectbox(
    "Alloy Type", ["All"] + available_materials
)

# Filter dataframe based on alloy selection
filtered_df = df.copy()
if selected_material != "All":
  filtered_df = filtered_df[
      filtered_df["material_name"].astype(str) == selected_material
  ]


def parse_porosity_value(val):
  # Extract leading numeric value from strings like "8.8±7.5" or return float
  if pd.isna(val):
    return None
  if isinstance(val, (int, float)):
    return float(val) if pd.notna(val) else None
  match = re.search(r"([+-]?\d+(?:\.\d+)?)", str(val))
  return float(match.group(1)) if match else None


plot_df = filtered_df.copy()
for num_col in [
    "laser_power_w",
    "scan_speed_mm_s",
    "hatch_distance_mm",
    "layer_thickness_mm",
    "spot_size_mm",
    "d50_um",
    "microhardness_hv",
]:
  if num_col in plot_df.columns:
    plot_df[num_col] = pd.to_numeric(plot_df[num_col], errors="coerce")

if "porosity_pct" in plot_df.columns:
  plot_df["porosity_num"] = plot_df["porosity_pct"].apply(parse_porosity_value)
else:
  plot_df["porosity_num"] = pd.NA

# Section B: Process Parameter Simulator (ML Input)
st.sidebar.subheader("2. Target Process Simulator")

sim_material = st.sidebar.selectbox("Simulator Alloy", available_materials)
material_data = df[df["material_name"].astype(str) == sim_material]

# Calculate safe default values using medians
default_power = _safe_median(material_data.get("laser_power_w", pd.Series()), 220.0)
default_speed = _safe_median(material_data.get("scan_speed_mm_s", pd.Series()), 900.0)
default_hatch = _safe_median(material_data.get("hatch_distance_mm", pd.Series()), 0.10)
default_layer = _safe_median(material_data.get("layer_thickness_mm", pd.Series()), 0.03)
default_spot = _safe_median(material_data.get("spot_size_mm", pd.Series()), 0.08)

if st.sidebar.button("Reset to default values"):
  st.session_state["sim_power"] = float(default_power)
  st.session_state["sim_speed"] = float(default_speed)
  st.session_state["sim_hatch"] = float(default_hatch)
  st.session_state["sim_layer"] = float(default_layer)
  st.session_state["sim_spot"] = float(default_spot)

sim_power = st.sidebar.slider(
    "Laser Power P (W)",
    min_value=30.0,
    max_value=600.0,
    value=float(default_power),
    step=1.0,
    key="sim_power",
)

sim_speed = st.sidebar.slider(
    "Scan Speed v (mm/s)",
    min_value=100.0,
    max_value=3500.0,
    value=float(default_speed),
    step=1.0,
    key="sim_speed",
)

sim_hatch = st.sidebar.number_input(
    "Hatch Distance h (mm)",
    min_value=0.001,
    max_value=1.0,
    value=float(default_hatch),
    step=0.001,
    format="%.3f",
    key="sim_hatch",
)

sim_layer = st.sidebar.number_input(
    "Layer Thickness t (mm)",
    min_value=0.001,
    max_value=1.0,
    value=float(default_layer),
    step=0.001,
    format="%.3f",
    key="sim_layer",
)

sim_spot = st.sidebar.number_input(
    "Laser Spot Size (mm)",
    min_value=0.001,
    max_value=1.0,
    value=float(default_spot),
    step=0.001,
    format="%.3f",
    key="sim_spot",
)

# 4. Machine Learning Model Inference
base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / "models" / "porosity_xgb_model.pkl"
used_ml_model = False

try:
  artifact = joblib.load(model_path)
  if isinstance(artifact, dict) and "model" in artifact:
    ml_model = artifact["model"]
    feature_cols = artifact.get(
        "features",
        [
            "laser_power_w",
            "scan_speed_mm_s",
            "hatch_distance_mm",
            "layer_thickness_mm",
            "spot_size_mm",
        ],
    )
    medians_map = artifact.get("feature_medians", {})
  else:
    ml_model = artifact
    feature_cols = [
        "laser_power_w",
        "scan_speed_mm_s",
        "hatch_distance_mm",
        "layer_thickness_mm",
        "spot_size_mm",
    ]
    medians_map = {}

  candidate_values = {
      "laser_power_w": sim_power,
      "scan_speed_mm_s": sim_speed,
      "hatch_distance_mm": sim_hatch,
      "layer_thickness_mm": sim_layer,
      "spot_size_mm": sim_spot,
  }
  row_dict = {
      col: candidate_values.get(col, medians_map.get(col, 0.0))
      for col in feature_cols
  }
  input_features = pd.DataFrame([row_dict])[feature_cols]
  pred_porosity = float(ml_model.predict(input_features)[0])
  used_ml_model = True
except Exception:
  # Heuristic fallback when trained model file is not yet available
  ratio_term = (sim_speed / max(sim_power, 1.0)) * (sim_hatch / max(sim_layer, 0.001))
  pred_porosity = max(0.02, min(5.0, float(0.15 * ratio_term)))

# 5. Main Workspace Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Interactive Dashboard & ML Prediction",
    "🔍 Literature Data Explorer",
    "🗺️ Process Window View",
    "📊 Material Comparison",
])

with tab1:
  st.subheader("Scan Speed vs. Laser Power (Porosity Color Map + Simulation)")
  if used_ml_model:
    st.success(f"✅ {used_ml_model} trained ML model used for porosity prediction.")
  else:
    st.warning(f"⚠️ {used_ml_model} trained ML model not found. Using heuristic fallback. ")
  col1, col2, col3 = st.columns(3)
  col1.metric(
      label="Simulated Scan Speed / Power",
      value=f"{sim_speed:.0f} mm/s | {sim_power:.0f} W",
  )

  col2.metric(
      label="ML Predicted Porosity",
      value=f"{pred_porosity:.3f} %",
      delta=(
          "Low-Porosity Ref (<0.5%)"
          if pred_porosity < 0.5
          else "Elevated Porosity Risk (>=0.5%)"
      ),
      delta_color="inverse",
  )

  col3.metric(
      label="Selected Alloy Profile",
      value=sim_material,
      help=(
          "Using trained ML model"
          if used_ml_model
          else "Using fallback heuristic estimate"
      ),
  )

  st.markdown("---")

  scatter_df = plot_df.dropna(
      subset=["scan_speed_mm_s", "laser_power_w", "porosity_num"]
  ).copy()

  desired_hover = [
      "publication_id",
      "material_name",
      "hatch_distance_mm",
      "layer_thickness_mm",
      "porosity_pct",
      "microhardness_hv",
  ]
  safe_hover_data = [c for c in desired_hover if c in scatter_df.columns]

  if not scatter_df.empty:
    fig = px.scatter(
        scatter_df,
        x="scan_speed_mm_s",
        y="laser_power_w",
        color="porosity_num",
        color_continuous_scale="Turbo",
        hover_data=safe_hover_data,
        title=(
            "Process Space (Scan Speed vs. Laser Power) Colored by Porosity (%)"
        ),
        labels={
            "scan_speed_mm_s": "Scan Speed (mm/s)",
            "laser_power_w": "Laser Power (W)",
            "porosity_num": "Porosity (%)",
        },
    )

    fig.update_traces(
        selector=dict(type="scatter"),
        marker=dict(size=9, opacity=0.82, line=dict(width=0.6, color="white")),
        showlegend=False,
    )

    # Overlay user's simulated target as a prominent red star with a clear legend entry
    fig.add_scatter(
        x=[sim_speed],
        y=[sim_power],
        mode="markers",
        marker=dict(
            size=22,
            color="red",
            symbol="star",
            line=dict(width=2, color="white"),
        ),
        name=f"Simulation Target ({sim_material}: {pred_porosity:.2f}%)",
        hovertemplate=(
            "<b>Simulation Target</b><br>"
            f"Alloy: {sim_material}<br>"
            "Scan Speed: %{x:.1f} mm/s<br>"
            "Laser Power: %{y:.1f} W<br>"
            f"Predicted Porosity: {pred_porosity:.3f}%<extra></extra>"
        ),
        showlegend=True,
    )

    fig.update_layout(
        height=540,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            title="Legend",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
  else:
    st.warning(
        "No valid records with Scan Speed, Laser Power, and Porosity match the"
        " current filter."
    )

with tab2:
  st.subheader("Curated Literature Data Explorer")
  st.markdown(
      "Browse, filter, and inspect experimental records extracted from"
      " literature."
  )
  st.dataframe(plot_df, use_container_width=True)

with tab3:
  st.subheader("Process Window Density Contours")
  st.markdown(
      "Process parameter mapping across Scan Speed and Laser Power colored by"
      " average porosity."
  )
  heatmap_df = plot_df.dropna(
      subset=["scan_speed_mm_s", "laser_power_w", "porosity_num"]
  )
  if len(heatmap_df) >= 3:
    fig_contour = px.density_heatmap(
        heatmap_df,
        x="scan_speed_mm_s",
        y="laser_power_w",
        z="porosity_num",
        histfunc="avg",
        color_continuous_scale="Viridis",
        title="Process Window (Scan Speed vs. Laser Power vs. Avg Porosity)",
        labels={
            "scan_speed_mm_s": "Scan Speed (mm/s)",
            "laser_power_w": "Laser Power (W)",
            "color": "Avg Porosity (%)",
        },
    )
    fig_contour.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_contour, use_container_width=True)
  else:
    st.info(
        "Insufficient valid data points in current filter to generate process"
        " window mapping."
    )

with tab4:
  st.subheader("Material Comparison & Statistical Analysis")
  st.markdown(
      "Compare manufacturing parameters of 316L and 18Ni300 using descriptive"
      " statistics from the literature dataset."
  )

  comparison_columns = [
      "laser_power_w",
      "scan_speed_mm_s",
      "hatch_distance_mm",
      "layer_thickness_mm",
      "spot_size_mm",
  ]
  available_comp_cols = [c for c in comparison_columns if c in df.columns]

  comparison_df = df.copy()
  for column in available_comp_cols:
    comparison_df[column] = pd.to_numeric(comparison_df[column], errors="coerce")

  if available_comp_cols and not comparison_df.empty:
    comparison_stats = (
        comparison_df.groupby("material_name")[available_comp_cols]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(3)
    )
    st.markdown("### Descriptive Statistics")
    st.dataframe(comparison_stats, use_container_width=True)

    st.markdown("---")
    st.markdown("### Interactive Parameter Comparison")

    parameter_labels = {
        "laser_power_w": "Laser Power (W)",
        "scan_speed_mm_s": "Scan Speed (mm/s)",
        "hatch_distance_mm": "Hatch Distance (mm)",
        "layer_thickness_mm": "Layer Thickness (mm)",
        "spot_size_mm": "Spot Size (mm)",
    }
    valid_options = [k for k in parameter_labels if k in available_comp_cols]

    selected_parameter = st.selectbox(
        "Select parameter to compare",
        options=valid_options,
        format_func=lambda x: parameter_labels[x],
    )

    fig_box = px.box(
        comparison_df,
        x="material_name",
        y=selected_parameter,
        color="material_name",
        points="all",
        title=f"{parameter_labels[selected_parameter]}: 316L vs 18Ni300",
        labels={
            "material_name": "Material",
            selected_parameter: parameter_labels[selected_parameter],
        },
    )
    fig_box.update_layout(
        height=500,
        template="plotly_white",
        showlegend=True,
    )
    st.plotly_chart(fig_box, use_container_width=True)
  else:
    st.info("No numeric process parameters available for material comparison.")