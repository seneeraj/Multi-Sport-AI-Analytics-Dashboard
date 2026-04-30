import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
# LOAD MODEL SAFELY
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
# ⚽ FOOTBALL SECTION
# ======================
if sport == "Football (EPL)":

    st.subheader("⚽ Football Player Performance")

    if model_f is not None:

        file_path = "data/football_players.csv"

        if os.path.exists(file_path):

            df_players = pd.read_csv(file_path)

            player = st.selectbox("Select Player", df_players["name"])

            player_data = df_players[df_players["name"] == player].iloc[0]

            goals = int(player_data["goals"])
            assists = int(player_data["assists"])
            minutes = int(player_data["minutes"])
            influence = float(player_data["influence"])
            creativity = float(player_data["creativity"])
            threat = float(player_data["threat"])
            ict = float(player_data["ict"])
            form = float(player_data["form"])
            bps = int(player_data["bps"])
            clean = int(player_data["clean"])
            conceded = int(player_data["conceded"])

            st.success(f"Loaded stats for {player}")

        else:
            st.warning("football_players.csv not found → using manual input")

            goals = st.number_input("Goals", 0, 50, value=5)
            assists = st.number_input("Assists", 0, 30, value=3)
            minutes = st.number_input("Minutes", 0, 4000, value=1500)
            influence = st.number_input("Influence", 0.0, 1000.0, value=300.0)
            creativity = st.number_input("Creativity", 0.0, 1000.0, value=250.0)
            threat = st.number_input("Threat", 0.0, 1000.0, value=280.0)
            ict = st.number_input("ICT Index", 0.0, 1000.0, value=300.0)
            form = st.number_input("Form", 0.0, 20.0, value=5.0)
            bps = st.number_input("BPS", 0, 1000, value=200)
            clean = st.number_input("Clean Sheets", 0, 50, value=5)
            conceded = st.number_input("Goals Conceded", 0, 100, value=10)

        if st.button("Predict Football Score"):

            try:
                features = np.array([[
                    goals, assists, minutes, influence, creativity,
                    threat, ict, form, bps, clean, conceded
                ]], dtype=float)

                prediction = model_f.predict(features)[0]

                st.success(f"🎯 Predicted Score: {round(prediction, 2)}")

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
                st.error(f"Error: {e}")

# ======================
# 🏏 CRICKET SECTION
# ======================
elif sport == "Cricket (IPL)":

    st.subheader("🏏 Cricket Player Performance")

    if model_c is not None:

        file_path = "data/player_stats.csv"

        if os.path.exists(file_path):

            df_players = pd.read_csv(file_path)

            player = st.selectbox("Select Player", df_players["batsman"])

            player_data = df_players[df_players["batsman"] == player].iloc[0]

            runs = float(player_data["batsman_runs"])
            strike_rate = float(player_data["strike_rate"])
            wickets = float(player_data["wickets"])
            fours = float(player_data["is_four"])
            sixes = float(player_data["is_six"])

            st.success(f"Loaded stats for {player}")

        else:
            st.warning("player_stats.csv not found → using manual input")

            runs = st.number_input("Runs", 0, 5000, value=500)
            strike_rate = st.number_input("Strike Rate", 0.0, 300.0, value=130.0)
            wickets = st.number_input("Wickets", 0, 500, value=20)
            fours = st.number_input("Fours", 0, 500, value=40)
            sixes = st.number_input("Sixes", 0, 300, value=20)

        if st.button("Predict Cricket Impact"):

            try:
                features = np.array([[
                    runs, strike_rate, wickets, fours, sixes
                ]], dtype=float)

                prediction = model_c.predict(features)[0]

                st.success(f"🏏 Predicted Impact Score: {round(prediction, 2)}")

                st.subheader("📊 Feature Importance")

                df_feat = pd.DataFrame({
                    "Feature": ["Runs", "Strike Rate", "Wickets", "Fours", "Sixes"],
                    "Importance": model_c.coef_
                })

                st.bar_chart(df_feat.set_index("Feature"))

            except Exception as e:
                st.error(f"Error: {e}")

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Multi-Sport AI Analytics System")
