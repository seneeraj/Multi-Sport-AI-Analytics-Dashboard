import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Sports AI Dashboard", layout="wide")

# ======================
# CUSTOM CSS (PREMIUM UI)
# ======================
st.markdown("""
<style>
.metric-card {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
}
.title {
    text-align:center;
    font-size:35px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODELS
# ======================
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Missing model: {path}")
        return None
    return pickle.load(open(path, "rb"))

model_f = load_model("models/football_model.pkl")
model_c = load_model("models/cricket_model.pkl")

# ======================
# HEADER
# ======================
st.markdown('<p class="title">⚽🏏 Multi-Sport AI Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

sport = st.sidebar.selectbox("Select Sport", ["Football (EPL)", "Cricket (IPL)"])

# ======================
# FOOTBALL
# ======================
if sport == "Football (EPL)":

    file_path = "data/football_players.csv"
    if not os.path.exists(file_path):
        st.error("football_players.csv missing")
        st.stop()

    df = pd.read_csv(file_path)

    # Fix column names
    df.rename(columns={
        "goals_scored": "goals",
        "ict_index": "ict",
        "clean_sheets": "clean",
        "goals_conceded": "conceded"
    }, inplace=True)

    st.header("⚽ Football Analytics")

    player = st.selectbox("Select Player", df["name"])
    row = df[df["name"] == player].iloc[0]

    goals = row["goals"]
    assists = row["assists"]
    influence = row["influence"]

    # KPI CARDS
    col1, col2, col3 = st.columns(3)
    col1.metric("Goals", int(goals))
    col2.metric("Assists", int(assists))
    col3.metric("Influence", round(influence,1))

    # Prediction
    if st.button("Predict Score"):
        features = np.array([[row["goals"], row["assists"], row["minutes"],
                              row["influence"], row["creativity"], row["threat"],
                              row["ict"], row["form"], row["bps"],
                              row["clean"], row["conceded"]]])

        pred = model_f.predict(features)[0]
        st.success(f"🎯 Predicted Score: {round(pred,2)}")

    avg = df.mean(numeric_only=True)

    # Comparison Chart
    st.subheader("📊 Player vs Average")
    comp_df = pd.DataFrame({
        "Player": [goals, assists, row["influence"], row["creativity"], row["threat"]],
        "Average": [
            avg["goals"], avg["assists"],
            avg["influence"], avg["creativity"], avg["threat"]
        ]
    }, index=["Goals","Assists","Influence","Creativity","Threat"])

    st.bar_chart(comp_df)

    # Percentile
    st.subheader("📊 Ranking")
    pct = (df["goals"] < goals).mean() * 100
    rank = df["goals"].rank(ascending=False, method="min")
    player_rank = int(rank[df["name"] == player].values[0])

    col1, col2 = st.columns(2)
    col1.metric("Percentile", f"{round(pct,1)}%")
    col2.metric("Rank", f"{player_rank}/{len(df)}")

    if pct > 85:
        st.success("🔥 Elite Player")
    elif pct > 60:
        st.info("⭐ Good Player")
    else:
        st.warning("📈 Developing Player")

    # Player Comparison
    st.markdown("---")
    st.subheader("⚔️ Player Comparison")

    p1_name = st.selectbox("Player 1", df["name"], key="f1")
    p2_name = st.selectbox("Player 2", df["name"], key="f2")

    p1 = df[df["name"] == p1_name].iloc[0]
    p2 = df[df["name"] == p2_name].iloc[0]

    col1, col2 = st.columns(2)
    col1.metric(p1_name, int(p1["goals"]))
    col2.metric(p2_name, int(p2["goals"]))

    comp = pd.DataFrame({
        p1_name: [p1["goals"], p1["assists"], p1["influence"]],
        p2_name: [p2["goals"], p2["assists"], p2["influence"]]
    }, index=["Goals","Assists","Influence"])

    st.line_chart(comp)

# ======================
# CRICKET
# ======================
elif sport == "Cricket (IPL)":

    file_path = "data/player_stats.csv"
    if not os.path.exists(file_path):
        st.error("player_stats.csv missing")
        st.stop()

    df = pd.read_csv(file_path)

    st.header("🏏 Cricket Analytics")

    player = st.selectbox("Select Player", df["batsman"])
    row = df[df["batsman"] == player].iloc[0]

    runs = row["batsman_runs"]
    sr = row["strike_rate"]

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Runs", int(runs))
    col2.metric("Strike Rate", round(sr,1))
    col3.metric("Wickets", int(row["wickets"]))

    # Prediction
    if st.button("Predict Impact"):
        features = np.array([[runs, sr, row["wickets"], row["is_four"], row["is_six"]]])
        pred = model_c.predict(features)[0]
        st.success(f"🏏 Impact Score: {round(pred,2)}")

    avg = df.mean(numeric_only=True)

    # Comparison
    st.subheader("📊 Player vs Average")
    comp_df = pd.DataFrame({
        "Player": [runs, sr, row["wickets"]],
        "Average": [
            avg["batsman_runs"], avg["strike_rate"], avg["wickets"]
        ]
    }, index=["Runs","Strike Rate","Wickets"])

    st.bar_chart(comp_df)

    # Percentile
    pct = (df["batsman_runs"] < runs).mean() * 100
    rank = df["batsman_runs"].rank(ascending=False, method="min")
    player_rank = int(rank[df["batsman"] == player].values[0])

    col1, col2 = st.columns(2)
    col1.metric("Percentile", f"{round(pct,1)}%")
    col2.metric("Rank", f"{player_rank}/{len(df)}")

    # Player Comparison
    st.markdown("---")
    st.subheader("⚔️ Player Comparison")

    p1_name = st.selectbox("Player 1", df["batsman"], key="c1")
    p2_name = st.selectbox("Player 2", df["batsman"], key="c2")

    p1 = df[df["batsman"] == p1_name].iloc[0]
    p2 = df[df["batsman"] == p2_name].iloc[0]

    comp = pd.DataFrame({
        p1_name: [p1["batsman_runs"], p1["strike_rate"], p1["wickets"]],
        p2_name: [p2["batsman_runs"], p2["strike_rate"], p2["wickets"]]
    }, index=["Runs","Strike Rate","Wickets"])

    st.line_chart(comp)
