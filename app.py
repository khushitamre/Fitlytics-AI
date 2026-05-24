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
# DATA + MODEL LOAD
# =========================
df = pd.read_csv("gym.csv")
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# =========================
# HERO HEADER (PREMIUM LOOK)
# =========================
st.markdown("""
<div style="
text-align:center;
padding:25px;
background:linear-gradient(90deg,#00ffe5,#0066ff);
border-radius:15px;
color:black;
font-size:30px;
font-weight:bold;">
💎 Fitlytics AI Pro Max
<br>
<span style="font-size:15px;">Next Generation Fitness Intelligence Dashboard</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.markdown("## 🎛 Smart Filters")
st.sidebar.markdown("---")

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

df = df[
    (df["Workout_Type"].isin(workout_filter)) &
    (df["Gender"].isin(gender_filter))
]

# =========================
# METRICS (CARDS STYLE)
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Records")
    st.markdown(f"<h2 style='color:#00ffe5'>{len(df)}</h2>", unsafe_allow_html=True)

with col2:
    st.markdown("### 🔥 Avg Calories")
    st.markdown(f"<h2 style='color:#00ffe5'>{round(df['Calories_Burned'].mean(),2)}</h2>", unsafe_allow_html=True)

with col3:
    st.markdown("### ⚡ Max Calories")
    st.markdown(f"<h2 style='color:#00ffe5'>{df['Calories_Burned'].max()}</h2>", unsafe_allow_html=True)

st.markdown("---")

# =========================
# GRAPHS (SIDE BY SIDE)
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("🏋️ Workout Impact on Calories")
    fig1 = px.box(df, x="Workout_Type", y="Calories_Burned", color="Workout_Type")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("❤️ Heart Rate Analysis")
    fig2 = px.scatter(df, x="Avg_BPM", y="Calories_Burned", color="Gender")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# =========================
# AI PREDICTION SYSTEM
# =========================
st.subheader("🔥 AI Calorie Predictor")

age = st.number_input("Age")
weight = st.number_input("Weight (kg)")
height = st.number_input("Height (m)")
bpm = st.slider("Avg BPM", 60, 200, 120)

if st.button("Predict Calories"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Weight (kg)": weight,
        "Height (m)": height,
        "Avg_BPM": bpm
    }])

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    result = model.predict(input_df)[0]

    st.success(f"🔥 Estimated Calories Burned: {result:.2f}")

    if result < 500:
        st.info("⚡ Light workout detected")
    elif result < 1000:
        st.success("💪 Moderate workout")
    else:
        st.warning("🔥 High intensity training")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Data Science Project | Fitlytics AI")
