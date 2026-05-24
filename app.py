import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================
# PAGE CONFIG (DARK STYLE)
# =========================
st.set_page_config(
    page_title="Fitlytics AI Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD DATA + MODEL
# =========================
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
df = pd.read_csv("gym.csv")

# =========================
# DARK THEME CSS (NEON LOOK)
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3 {
    color: #00ffe5;
}
.stButton>button {
    background-color: #00ffe5;
    color: black;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("💎 Fitlytics AI Pro Max")
st.markdown("⚡ Advanced AI Fitness Analytics Dashboard")

st.markdown("---")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.title("🎛 Filters")

workout_filter = st.sidebar.multiselect(
    "Workout Type",
    df["Workout_Type"].unique(),
    default=df["Workout_Type"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    df["Gender"].unique(),
    default=df["Gender"].unique()
)

filtered_df = df[
    (df["Workout_Type"].isin(workout_filter)) &
    (df["Gender"].isin(gender_filter))
]

# =========================
# KPI CARDS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Records", len(filtered_df))
col2.metric("Avg Calories", round(filtered_df["Calories_Burned"].mean(), 2))
col3.metric("Max Calories", filtered_df["Calories_Burned"].max())

st.markdown("---")

# =========================
# 3D ANIMATED GRAPH
# =========================
st.subheader("📊 3D Workout Intelligence")

fig = px.scatter_3d(
    filtered_df,
    x="Session_Duration (hours)",
    y="Avg_BPM",
    z="Calories_Burned",
    color="Workout_Type",
    size="Calories_Burned",
    hover_data=["Age", "Gender"],
    title="Fitlytics AI 3D Engine"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =========================
# PREDICTION SYSTEM
# =========================
st.subheader("🔥 AI Calorie Predictor")

age = st.number_input("Age")
weight = st.number_input("Weight (kg)")
height = st.number_input("Height (m)")
bpm = st.slider("Avg BPM", 60, 200, 120)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Avg_BPM": bpm
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    result = model.predict(input_df)[0]

    st.success(f"🔥 Estimated Calories: {result:.2f}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Built with AI | Fitlytics Pro Max | Data Science Project")
