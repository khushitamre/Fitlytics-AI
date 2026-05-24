import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fitlytics AI Pro Max",
    layout="wide"
)

# =========================
# LOAD DATA + MODEL
# =========================
df = pd.read_csv("gym.csv")

model = joblib.load("model.pkl")

# =========================
# PREMIUM HEADER
# =========================
st.markdown("""
<div style="
text-align:center;
padding:25px;
background:linear-gradient(90deg,#00ffe5,#0066ff);
border-radius:18px;
color:black;
font-size:34px;
font-weight:bold;">
💎 Fitlytics AI Pro Max
<br>
<span style="font-size:18px;">
Next Generation Fitness Intelligence Dashboard
</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🎛 Smart Dashboard Filters")

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
# METRICS
# =========================
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "📊 Total Records",
        len(filtered_df)
    )

with c2:
    st.metric(
        "🔥 Avg Calories",
        round(filtered_df["Calories_Burned"].mean(), 2)
    )

with c3:
    st.metric(
        "⚡ Max Calories",
        round(filtered_df["Calories_Burned"].max(), 2)
    )

st.markdown("---")

# =========================
# CHARTS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏋️ Workout Impact")

    fig1 = px.box(
        filtered_df,
        x="Workout_Type",
        y="Calories_Burned",
        color="Workout_Type"
    )

    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("❤️ Heart Rate Analysis")

    fig2 = px.scatter(
        filtered_df,
        x="Avg_BPM",
        y="Calories_Burned",
        color="Gender",
        size="Session_Duration (hours)"
    )

    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# =========================
# AI PREDICTOR
# =========================
st.subheader("🤖 AI Calorie Burn Predictor")

a1, a2, a3 = st.columns(3)

with a1:
    age = st.number_input("Age", 10, 80, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.75)

with a2:
    max_bpm = st.slider("Max BPM", 100, 220, 180)
    avg_bpm = st.slider("Avg BPM", 60, 200, 140)
    resting_bpm = st.slider("Resting BPM", 40, 100, 65)
    duration = st.slider("Workout Duration (hours)", 0.5, 5.0, 1.5)

with
