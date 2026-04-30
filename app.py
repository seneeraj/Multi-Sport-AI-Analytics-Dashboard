import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Multi-Sport AI Dashboard", layout="wide")

# ======================
# LOAD MODEL SAFELY
# ======================
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        return None
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_f = load_model("models/football_model.pkl")
model_c = load_model("models/cricket_model.pkl")

# ======================
# HEADER
# ======================
st.title("⚽🏏 Multi-Sport AI Analytics Dashboard")
st.markdown("---")

sport = st.sidebar.selectbox("Select Sport", ["Football (EPL)", "Cricket (IPL)"])

# ======================
# FOOTBALL SECTION
# ======================
if sport == "Football (EPL)":

    st.header("⚽ Football Analytics")

    file_path = "data/football_players.csv"

    if not os.path.exists(file_path):
        st.error("football_players.csv not found")
        st.stop()

    df = pd.read_csv(file_path)

    # 🔥 Fix column mismatch
    df.rename(columns={
        "goals_scored": "goals",
        "ict_index": "ict",
        "clean_sheets": "clean",
        "goals_conceded": "conceded"
    }, inplace=True)

    player = st.selectbox("Select Player", df["name"])
    row = df[df["name"] == player].iloc[0]

    goals = float(row["goals"])
    assists = float(row["assists"])
    minutes = float(row["minutes"])
    influence = float(row["influence"])
    creativity = float(row["creativity"])
    threat = float(row["threat"])
    ict = float(row["ict"])
    form = float(row["form"])
    bps = float(row["bps"])
    clean = float(row["clean"])
    conceded = float(row["conceded"])

    # Prediction
    if st.button("Predict Football Score"):
        features = np.array([[goals, assists, minutes, influence, creativity,
                              threat, ict, form, bps, clean, conceded]])

        pred = model_f.predict(features)[0]
        st.success(f"🎯 Predicted Score: {round(pred,2)}")

    # ======================
    # VISUALS
    # ======================

    avg = df.mean(numeric_only=True)

    st.subheader("📊 Player vs Average")
    comp_df = pd.DataFrame({
        "Player": [goals, assists, influence, creativity, threat],
        "Average": [
            avg["goals"], avg["assists"],
            avg["influence"], avg["creativity"], avg["threat"]
        ]
    }, index=["Goals","Assists","Influence","Creativity","Threat"])

    st.bar_chart(comp_df)

    # Radar alternative (line chart)
    st.subheader("🕸️ Performance Trend")
    radar_df = pd.DataFrame({
        "Feature": ["Goals","Assists","Influence","Creativity","Threat"],
        "Player": [goals, assists, influence, creativity, threat],
        "Average": [
            avg["goals"], avg["assists"],
            avg["influence"], avg["creativity"], avg["threat"]
        ]
    }).set_index("Feature")

    st.line_chart(radar_df)

    # Leaderboard
    st.subheader("🏆 Top Players")
    st.dataframe(df.sort_values("goals", ascending=False).head(10))

    # Distribution
    st.subheader("📈 Goals Distribution")
    st.bar_chart(df["goals"].value_counts())

    # ======================
    # PERCENTILE + RANK
    # ======================

    st.subheader("📊 Ranking & Percentile")

    goals_pct = (df["goals"] < goals).mean() * 100
    assists_pct = (df["assists"] < assists).mean() * 100
    influence_pct = (df["influence"] < influence).mean() * 100

    overall_pct = np.mean([goals_pct, assists_pct, influence_pct])

    rank = df["goals"].rank(ascending=False, method="min")
    player_rank = int(rank[df["name"] == player].values[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Percentile", f"{round(overall_pct,1)}%")
    col2.metric("Rank", f"{player_rank} / {len(df)}")
    col3.metric("Goals Percentile", f"{round(goals_pct,1)}%")

    if overall_pct > 85:
        st.success("🔥 Elite Player")
    elif overall_pct > 60:
        st.info("⭐ Good Player")
    else:
        st.warning("📈 Developing Player")

# ======================
# CRICKET SECTION
# ======================
elif sport == "Cricket (IPL)":

    st.header("🏏 Cricket Analytics")

    file_path = "data/player_stats.csv"

    if not os.path.exists(file_path):
        st.error("player_stats.csv not found")
        st.stop()

    df = pd.read_csv(file_path)

    player = st.selectbox("Select Player", df["batsman"])
    row = df[df["batsman"] == player].iloc[0]

    runs = float(row["batsman_runs"])
    strike_rate = float(row["strike_rate"])
    wickets = float(row["wickets"])
    fours = float(row["is_four"])
    sixes = float(row["is_six"])

    # Prediction
    if st.button("Predict Cricket Impact"):
        features = np.array([[runs, strike_rate, wickets, fours, sixes]])
        pred = model_c.predict(features)[0]
        st.success(f"🏏 Predicted Impact Score: {round(pred,2)}")

    # ======================
    # VISUALS
    # ======================

    avg = df.mean(numeric_only=True)

    st.subheader("📊 Player vs Average")
    comp_df = pd.DataFrame({
        "Player": [runs, strike_rate, wickets, fours, sixes],
        "Average": [
            avg["batsman_runs"], avg["strike_rate"],
            avg["wickets"], avg["is_four"], avg["is_six"]
        ]
    }, index=["Runs","Strike Rate","Wickets","Fours","Sixes"])

    st.bar_chart(comp_df)

    # Radar alternative
    st.subheader("🕸️ Performance Trend")
    radar_df = pd.DataFrame({
        "Feature": ["Runs","Strike Rate","Wickets","Fours","Sixes"],
        "Player": [runs, strike_rate, wickets, fours, sixes],
        "Average": [
            avg["batsman_runs"], avg["strike_rate"],
            avg["wickets"], avg["is_four"], avg["is_six"]
        ]
    }).set_index("Feature")

    st.line_chart(radar_df)

    # Leaderboard
    st.subheader("🏆 Top Players")
    st.dataframe(df.sort_values("batsman_runs", ascending=False).head(10))

    # Distribution
    st.subheader("📈 Runs Distribution")
    st.bar_chart(df["batsman_runs"].value_counts())

    # ======================
    # PERCENTILE + RANK
    # ======================

    st.subheader("📊 Ranking & Percentile")

    runs_pct = (df["batsman_runs"] < runs).mean() * 100
    sr_pct = (df["strike_rate"] < strike_rate).mean() * 100
    wicket_pct = (df["wickets"] < wickets).mean() * 100

    overall_pct = np.mean([runs_pct, sr_pct, wicket_pct])

    rank = df["batsman_runs"].rank(ascending=False, method="min")
    player_rank = int(rank[df["batsman"] == player].values[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Percentile", f"{round(overall_pct,1)}%")
    col2.metric("Rank", f"{player_rank} / {len(df)}")
    col3.metric("Runs Percentile", f"{round(runs_pct,1)}%")

    if overall_pct > 85:
        st.success("🔥 Elite Player")
    elif overall_pct > 60:
        st.info("⭐ Good Player")
    else:
        st.warning("📈 Developing Player")
