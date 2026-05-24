import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Fitlytics AI Pro Max",
    layout="wide"
)

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv("gym.csv")

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load("model.pkl")

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
    color: white;
}

.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}

.metric-card {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,255,255,0.3);
}

.big-font {
    font-size: 35px;
    font-weight: bold;
    color: #00ffe5;
}

.small-font {
    font-size: 16px;
    color: #cccccc;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================

st.markdown("""
<div style='text-align:center;padding:25px;border-radius:20px;
background: linear-gradient(90deg,#00ffe5,#00aaff);'>
<h1 style='color:black;'>💎 Fitlytics AI Pro Max</h1>
<p style='color:black;font-size:18px;'>
Next Generation Fitness Intelligence Dashboard
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🎛 Smart Filters")

gender = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

workout = st.sidebar.multiselect(
    "Select Workout",
    options=df["Workout_Type"].unique(),
    default=df["Workout_Type"].unique()
)

filtered_df = df[
    (df["Gender"].isin(gender)) &
    (df["Workout_Type"].isin(workout))
]

# ============================================
# METRICS
# ============================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
    <div class='small-font'>📊 Total Records</div>
    <div class='big-font'>{len(filtered_df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_cal = round(filtered_df["Calories_Burned"].mean(), 2)

    st.markdown(f"""
    <div class='metric-card'>
    <div class='small-font'>🔥 Average Calories</div>
    <div class='big-font'>{avg_cal}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    max_cal = filtered_df["Calories_Burned"].max()

    st.markdown(f"""
    <div class='metric-card'>
    <div class='small-font'>⚡ Maximum Calories</div>
    <div class='big-font'>{max_cal}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# CHARTS
# ============================================

c1, c2 = st.columns(2)

with c1:
    st.subheader("🏋 Workout vs Calories")

    fig1 = px.box(
        filtered_df,
        x="Workout_Type",
        y="Calories_Burned",
        color="Workout_Type"
    )

    st.plotly_chart(fig1, use_container_width=True)

with c2:
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

# ============================================
# AI CALORIE PREDICTOR
# ============================================

st.subheader("🔥 AI Calorie Predictor")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 15, 60, 25)

    weight = st.slider("Weight (kg)", 40, 120, 70)

    height = st.slider("Height (m)", 1.4, 2.2, 1.75)

    duration = st.slider("Workout Duration (hours)", 1.0, 3.0, 1.5)

with col2:

    avg_bpm = st.slider("Average BPM", 60, 200, 120)

    resting_bpm = st.slider("Resting BPM", 40, 100, 70)

    water = st.slider("Water Intake (liters)", 1.0, 5.0, 2.5)

    frequency = st.slider("Workout Frequency", 1, 7, 4)

gender_input = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

workout_input = st.selectbox(
    "Workout Type",
    ["Cardio", "HIIT", "Strength", "Yoga"]
)

if st.button("🚀 Predict Calories Burned"):

    input_data = pd.DataFrame({

        "Age": [age],
        "Gender": [gender_input],
        "Weight (kg)": [weight],
        "Height (m)": [height],
        "Avg_BPM": [avg_bpm],
        "Resting_BPM": [resting_bpm],
        "Session_Duration (hours)": [duration],
        "Workout_Type": [workout_input],
        "Water_Intake (liters)": [water],
        "Workout_Frequency (days/week)": [frequency]

    })

    input_data = pd.get_dummies(input_data)

    model_columns = model.feature_names_in_

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    st.success(f"🔥 Estimated Calories Burned: {prediction:.2f}")

    if prediction < 400:
        st.info("⚡ Light Workout Session")

    elif prediction < 800:
        st.success("💪 Moderate Workout")

    else:
        st.warning("🔥 High Intensity Training")

st.markdown("---")

# ============================================
# DATA TABLE
# ============================================

st.subheader("📋 Fitness Dataset Preview")

st.dataframe(filtered_df)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div style='text-align:center;padding:20px;color:white;'>
🚀 Built with Streamlit + Machine Learning <br>
💎 Premium Data Science Project by Khushi Tamre
</div>
""", unsafe_allow_html=True)
