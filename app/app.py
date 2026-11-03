import os
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
st.title("🔬 Metal LPBF Literature Database & Process ML Simulator")
st.markdown(
    "A research-focused MVP integrating curated literature data (600+ samples)"
    " with machine learning-driven process parameter prediction."
)


# 2. Data Loading Function with Caching
@st.cache_data
def load_literature_data():
  # Load the curated dataset
  csv_path = "data/raw_papers.csv"
  if os.path.exists(csv_path):
    return pd.read_csv(csv_path)
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
selected_material = st.sidebar.selectbox(
    "Alloy Type", ["All", "Ti-6Al-4V", "AlSi10Mg"]
)

# Filter dataframe based on alloy selection
filtered_df = df.copy()
if selected_material != "All":
  filtered_df = filtered_df[filtered_df["material_name"] == selected_material]

# Section B: Process Parameter Simulator (ML Input)
st.sidebar.subheader("2. Target Process Simulator")
sim_power = st.sidebar.slider(
    "Laser Power P (W)", min_value=100, max_value=500, value=300, step=10
)
sim_speed = st.sidebar.slider(
    "Scan Speed v (mm/s)", min_value=200, max_value=2000, value=1000, step=50
)
sim_hatch = st.sidebar.number_input(
    "Hatch Distance h (mm)", value=0.10, step=0.01, format="%.2f"
)
sim_layer = st.sidebar.number_input(
    "Layer Thickness t (mm)", value=0.03, step=0.005, format="%.3f"
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
tab1, tab2, tab3 = st.tabs([
    "📈 Interactive Dashboard & ML Prediction",
    "🔍 Literature Data Explorer",
    "🗺️ Process Window View",
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

  col3.metric(label="Selected Alloy Profile", value=selected_material)

  st.markdown("---")

  # Core Visualization: Literature Scatter Plot with Red Star Overlay for ML Prediction
  if not filtered_df.empty:
    fig = px.scatter(
        filtered_df,
        x="ved_j_mm3",
        y="porosity_pct",
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
            "porosity_pct": "Porosity (%)",
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
  st.dataframe(filtered_df, use_container_width=True)

with tab3:
  st.subheader("Process Window Density Contours")
  st.markdown(
      "Global process parameter mapping across Laser Power and Scan Speed."
  )
  if not filtered_df.empty and len(filtered_df) >= 3:
    # 修复方案：改用 px.density_heatmap，它完美支持 color_continuous_scale
    fig_contour = px.density_heatmap(
        filtered_df,
        x="laser_power_w",
        y="scan_speed_mm_s",
        z="porosity_pct",
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