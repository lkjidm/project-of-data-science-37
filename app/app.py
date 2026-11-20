import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import re

# 1. Page Configuration & Layout Setup
st.set_page_config(
    page_title="LPBF Research Portal & ML Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main Title
st.title("🔬 Metal LPBF Literature Database & ML Simulator")
st.markdown(
    "A research-focused MVP integrating curated literature data (600+ samples)"
    " with machine learning-driven process parameter prediction."
)


# 2. Data Loading Function with Caching
@st.cache_data
def load_literature_data():
  # Load the curated dataset
  BASE_DIR = Path(__file__).resolve().parent.parent
  csv_path = BASE_DIR / "data" / "raw_papers.csv"
  if os.path.exists(csv_path):
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
        "RD (%)": "relative_density_pct",
        "Printer Model": "printer_model",
        "DOI": "doi",
        "Energy density (J/mm³)": "ved_j_mm3",
        "Porosity in vol%": "porosity_pct",
        "Microhadness (HV)": "microhardness_hv",
        # "Hatch Angle(degree)": "hatch_angle_deg",
        # "Exposure time in µs": "exposure_time_us",
    }

    data = data.rename(columns=column_map)

    return data
  else:
    # Fallback mock data if csv is not yet created, ensuring app runs smoothly
    return pd.DataFrame({
        "publication_id": ["Smith_2023", "Wang_2024", "Chen_2022", "Li_2023"],
        "material_name": ["Ti-6Al-4V", "AlSi10Mg", "Ti-6Al-4V", "AlSi10Mg"],
        "laser_power_w": [250, 350, 200, 400],
        "scan_speed_mm_s": [800, 1200, 600, 1000],
        "hatch_distance_mm": [0.10, 0.12, 0.09, 0.10],
        "layer_thickness_mm": [0.03, 0.03, 0.03, 0.03],
        "ved_j_mm3": [104.17, 81.02, 123.46, 133.33],
        "porosity_pct": [0.05, 0.25, 0.02, 1.20],
        "microhardness_hv": [350, 130, 365, 125],
        "build_orientation": ["XY", "Z", "XY", "Z"],
    })


df = load_literature_data()

# 3. Sidebar: Control Panel (Filters & Simulator)
st.sidebar.header("🎛️ Control Panel")

# Section A: Global Literature Filters
st.sidebar.subheader("1. Literature Filter")
available_materials = sorted(df["material_name"].dropna().unique())

selected_material = st.sidebar.selectbox(
    "Alloy Type", ["All"] + available_materials
)

# Filter dataframe based on alloy selection
filtered_df = df.copy()
if selected_material != "All":
  filtered_df = filtered_df[filtered_df["material_name"] == selected_material]

def parse_porosity_value(val):
  # Extract leading numeric value from strings like "8.8±7.5" or return float
  if pd.isna(val):
    return None
  if isinstance(val, (int, float)):
    return float(val)
  match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)", str(val))
  return float(match.group(1)) if match else None


plot_df = filtered_df.copy()
plot_df["ved_j_mm3"] = pd.to_numeric(plot_df["ved_j_mm3"], errors="coerce")
plot_df["porosity_num"] = plot_df["porosity_pct"].apply(parse_porosity_value)
plot_df = plot_df.dropna(subset=["ved_j_mm3", "porosity_num"])

# Section B: Process Parameter Simulator (ML Input)
st.sidebar.subheader("2. Target Process Simulator")

# Select material for the simulator
sim_material = st.sidebar.selectbox(
    "Simulator Alloy", available_materials
)

# Get literature records for the selected material
material_data = df[df["material_name"] == sim_material]

# Calculate default values using the median of the dataset
default_power = material_data["laser_power_w"].median()
default_speed = material_data["scan_speed_mm_s"].median()
default_hatch = material_data["hatch_distance_mm"].median()
default_layer = material_data["layer_thickness_mm"].median()
default_spot = material_data["spot_size_mm"].median()

# Add a reset button
if st.sidebar.button("Reset to default values"):
    st.session_state["sim_power"] = float(default_power)
    st.session_state["sim_speed"] = float(default_speed)
    st.session_state["sim_hatch"] = float(default_hatch)
    st.session_state["sim_layer"] = float(default_layer)
    st.session_state["sim_spot"] = float(default_spot)

# Simulator inputs
sim_power = st.sidebar.slider(
    "Laser Power P (W)",
    min_value=30.0,
    max_value=500.0,
    value=float(default_power),
    step=1.0,
    key="sim_power"
)

sim_speed = st.sidebar.slider(
    "Scan Speed v (mm/s)",
    min_value=100.0,
    max_value=3500.0,
    value=float(default_speed),
    step=1.0,
    key="sim_speed"
)

sim_hatch = st.sidebar.number_input(
    "Hatch Distance h (mm)",
    min_value=0.001,
    max_value=1.0,
    value=float(default_hatch),
    step=0.001,
    format="%.3f",
    key="sim_hatch"
)

sim_layer = st.sidebar.number_input(
    "Layer Thickness t (mm)",
    min_value=0.001,
    max_value=1.0,
    value=float(default_layer),
    step=0.001,
    format="%.3f",
    key="sim_layer"
)
sim_spot = st.sidebar.number_input(
    "Laser Spot Size (mm)",
    min_value=0.001,
    max_value=1.0,
    value=float(default_spot),
    step=0.001,
    format="%.3f",
    key="sim_spot"
)
# Automatically calculate Volume Energy Density (VED) for simulation target
sim_ved = sim_power / (sim_speed * sim_hatch * sim_layer)

# 4. Machine Learning Model Inference
# Load trained XGBoost model if available, otherwise use a fallback calculation
model_path = "models/xgb_porosity.pkl"
try:
  ml_model = joblib.load(model_path)
  # Construct feature vector matching training schema
  input_features = pd.DataFrame(
      [[sim_power, sim_speed, sim_hatch, sim_layer, sim_ved]],
      columns=[
          "laser_power_w",
          "scan_speed_mm_s",
          "hatch_distance_mm",
          "layer_thickness_mm",
          "ved_j_mm3",
      ],
  )
  pred_porosity = ml_model.predict(input_features)[0]
except Exception:
  # Heuristic fallback for prototype testing if pkl is not yet compiled
  pred_porosity = max(
      0.01, min(5.0, float(120.0 / (sim_ved + 1.0) * 0.1))
  )

# 5. Main Workspace Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Interactive Dashboard & ML Prediction",
    "🔍 Literature Data Explorer",
    "🗺️ Process Window View",
    "📊 Material Comparison",
])

with tab1:
  st.subheader("Energy Density (VED) vs. Porosity (Literature Cloud + Simulation)")

  # Top Metric Cards displaying real-time simulation output
  col1, col2, col3 = st.columns(3)
  col1.metric(
      label="Calculated VED",
      value=f"{sim_ved:.2f} J/mm³",
      help="Volume Energy Density = P / (v * h * t)",
  )

  col2.metric(
      label="ML Predicted Porosity",
      value=f"{pred_porosity:.3f} %",
      delta=(
          "Safe Zone (<0.5%)" if pred_porosity < 0.5 else "High Porosity Risk"
      ),
      delta_color="inverse",
  )

  col3.metric(label="Selected Alloy Profile", value=sim_material)

  st.markdown("---")

  # Core Visualization: Literature Scatter Plot with Red Star Overlay for ML Prediction
  if not plot_df.empty:
    fig = px.scatter(
        plot_df,
        x="ved_j_mm3",
        y="porosity_num",
        color="material_name",
        hover_data=[
            "publication_id",
            "laser_power_w",
            "scan_speed_mm_s",
            "microhardness_hv",
        ],
        title=(
            "Literature Scatter Cloud with Real-Time ML Simulation Target (Red"
            " Star)"
        ),
        labels={
            "ved_j_mm3": "Volume Energy Density (J/mm³)",
            "porosity_num": "Porosity (%)",
            "material_name": "Alloy",
        },
    )

   
    # Overlay user's simulated target as a prominent red star marker
    fig.add_scatter(
        x=[sim_ved],
        y=[pred_porosity],
        mode="markers",
        marker=dict(
            size=22,
            color="red",
            symbol="star",
            line=dict(width=2, color="white"),
        ),
        name="Your Simulation Target",
    )

    fig.update_layout(height=520, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
  else:
    st.warning("No literature records match the selected filter criteria.")

with tab2:
  st.subheader("Curated Literature Data Explorer")
  st.markdown(
      "Browse, filter, and inspect the raw experimental data points extracted"
      " from peer-reviewed publications."
  )
  st.dataframe(plot_df, use_container_width=True)

with tab3:
  st.subheader("Process Window Density Contours")
  st.markdown(
      "Global process parameter mapping across Laser Power and Scan Speed."
  )
  if not plot_df.empty and len(plot_df) >= 3:
    # 修复方案：改用 px.density_heatmap，它完美支持 color_continuous_scale
    fig_contour = px.density_heatmap(
        plot_df,
        x="laser_power_w",
        y="scan_speed_mm_s",
        z="porosity_num",
        histfunc="avg",  # 对重叠区域取平均孔隙率
        color_continuous_scale="Viridis",
        title="Process Window (Power vs. Scan Speed vs. Porosity)",
        labels={
            "laser_power_w": "Laser Power (W)",
            "scan_speed_mm_s": "Scan Speed (mm/s)",
            "color": "Avg Porosity (%)",
        },
    )
    fig_contour.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_contour, use_container_width=True)
  else:
    st.info(
        "Insufficient data points in current filter to generate contour"
        " mapping."
    )

# 6. Material Comparison & Statistical Analysis
with tab4:
    st.subheader("Material Comparison & Statistical Analysis")

    st.markdown(
        "Compare the manufacturing parameters of 316L and 18Ni300 "
        "using descriptive statistics from the literature dataset."
    )

    comparison_columns = [
        "laser_power_w",
        "scan_speed_mm_s",
        "hatch_distance_mm",
        "layer_thickness_mm",
        "spot_size_mm",
        "ved_j_mm3",
    ]

    # Create a copy to avoid modifying the original dataset
    comparison_df = df.copy()

    # Ensure all comparison variables are numeric
    for column in comparison_columns:
        comparison_df[column] = pd.to_numeric(
            comparison_df[column], errors="coerce"
        )

    # Calculate descriptive statistics for each material
    comparison_stats = comparison_df.groupby("material_name")[
        comparison_columns
    ].agg(["count", "mean", "median", "std", "min", "max"])

    comparison_stats = comparison_stats.round(3)

    st.markdown("### Descriptive Statistics")

    st.markdown(
        "The table below shows the distribution of each manufacturing "
        "parameter across the two materials."
    )

    st.dataframe(
        comparison_stats,
        use_container_width=True
    )
    st.markdown("---")
    st.markdown("### Interactive Parameter Comparison")

    # Select manufacturing parameter
    parameter_labels = {
        "laser_power_w": "Laser Power (W)",
        "scan_speed_mm_s": "Scan Speed (mm/s)",
        "hatch_distance_mm": "Hatch Distance (mm)",
        "layer_thickness_mm": "Layer Thickness (mm)",
        "spot_size_mm": "Spot Size (mm)",
        "ved_j_mm3": "Volume Energy Density (J/mm³)",
    }

    selected_parameter = st.selectbox(
        "Select parameter to compare",
        options=list(parameter_labels.keys()),
        format_func=lambda x: parameter_labels[x],
    )

    # Create box plot
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
        showlegend=False,
    )

    st.plotly_chart(fig_box, use_container_width=True)