import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Multi-Sport AI Dashboard",
    layout="wide",
    page_icon="⚽"
)

# ======================
# SAFE MODEL LOADING
# ======================
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model not found: {path}")
        return None
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model_f = load_model("models/football_model.pkl")
model_c = load_model("models/cricket_model.pkl")

# ======================
# HEADER
# ======================
st.markdown("""
<h1 style='text-align:center; color:#2E86C1;'>
⚽🏏 Multi-Sport AI Performance Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================
# SIDEBAR
# ======================
sport = st.sidebar.selectbox(
    "Select Sport",
    ["Football (EPL)", "Cricket (IPL)"]
)

# ======================
# KPI CARD FUNCTION
# ======================
def kpi(title, value):
    st.metric(label=title, value=value)

# ======================
# ⚽ FOOTBALL SECTION
# ======================
if sport == "Football (EPL)":

    st.subheader("⚽ Football Player Performance Predictor")

    if model_f is not None:

        col1, col2, col3 = st.columns(3)

        with col1:
            goals = st.number_input("Goals", 0, 50)
            assists = st.number_input("Assists", 0, 30)
            minutes = st.number_input("Minutes", 0, 4000)

        with col2:
            influence = st.number_input("Influence", 0.0, 1000.0)
            creativity = st.number_input("Creativity", 0.0, 1000.0)
            threat = st.number_input("Threat", 0.0, 1000.0)

        with col3:
            ict = st.number_input("ICT Index", 0.0, 1000.0)
            form = st.number_input("Form", 0.0, 20.0)
            bps = st.number_input("BPS", 0, 1000)
            clean = st.number_input("Clean Sheets", 0, 50)
            conceded = st.number_input("Goals Conceded", 0, 100)

        if st.button("Predict Football Score"):

            try:
                features = np.array([[
                    goals,
                    assists,
                    minutes,
                    influence,
                    creativity,
                    threat,
                    ict,
                    form,
                    bps,
                    clean,
                    conceded
                ]], dtype=float)

                prediction = model_f.predict(features)[0]

                st.success(f"🎯 Predicted Score: {round(prediction, 2)}")

                # Feature importance
                st.subheader("📊 Feature Importance")

                df_feat = pd.DataFrame({
                    "Feature": [
                        "Goals", "Assists", "Minutes", "Influence",
                        "Creativity", "Threat", "ICT", "Form",
                        "BPS", "Clean Sheets", "Conceded"
                    ],
                    "Importance": model_f.coef_
                })

                st.bar_chart(df_feat.set_index("Feature"))

            except Exception as e:
                st.error(f"Error in prediction: {e}")

# ======================
# 🏏 CRICKET SECTION (UPDATED)
# ======================

elif sport == "Cricket (IPL)":

    st.subheader("🏏 Cricket Player Performance (Auto Mode)")

    if model_c is not None:

        # Load player stats
        df_players = pd.read_csv("data/player_stats.csv")

        # Dropdown
        player = st.selectbox("Select Player", df_players["batsman"].unique())

        # Get selected player data
        player_data = df_players[df_players["batsman"] == player].iloc[0]

        # Show stats
        col1, col2, col3 = st.columns(3)

        col1.metric("Runs", int(player_data["batsman_runs"]))
        col2.metric("Strike Rate", round(player_data["strike_rate"], 2))
        col3.metric("Wickets", int(player_data["wickets"]))

        col4, col5 = st.columns(2)

        col4.metric("Fours", int(player_data["is_four"]))
        col5.metric("Sixes", int(player_data["is_six"]))

        if st.button("Predict Player Impact"):

            try:
                features = [[
                    player_data["batsman_runs"],
                    player_data["strike_rate"],
                    player_data["wickets"],
                    player_data["is_four"],
                    player_data["is_six"]
                ]]

                features = np.array(features, dtype=float)

                prediction = model_c.predict(features)[0]

                st.success(f"🏏 Predicted Impact Score: {round(prediction, 2)}")

            except Exception as e:
                st.error(f"Error: {e}")
# ======================
# 📂 DATA EXPLORER
# ======================
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Dataset")

uploaded = st.sidebar.file_uploader("Upload CSV")

if uploaded:
    try:
        df = pd.read_csv(uploaded)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📈 Correlation Matrix")
        st.dataframe(df.corr(numeric_only=True))

    except Exception as e:
        st.error(f"Error reading file: {e}")
