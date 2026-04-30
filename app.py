import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Sports AI Dashboard", layout="wide")

st.markdown("""
<style>
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}
.sidebar-section {
    font-size: 14px;
    color: gray;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 🚀 PREMIUM SIDEBAR
# ======================

with st.sidebar:

    st.markdown("## ⚙️ Navigation")

    sport_ui = st.radio(
        "Select Sport",
        ["⚽ Football (EPL)", "🏏 Cricket (IPL)"],
        index=0,
        key="sport_selector"
    )

    # ✅ FIX: Map emoji label → original values (DO NOT CHANGE LOGIC BELOW)
    if "Football" in sport:
        sport = "Football (EPL)"
    elif "Cricket" in sport:
        sport = "Cricket (IPL)"

    st.markdown("---")

    # 🎥 GIF / Visual Section
    st.markdown("## 🎬 Insights Zone")

    st.image(
        "https://assets.sportsboom.com/England_bowler_Joe_Root_bowls_to_batsman_Mohammed_Siraj_3c07c8dad8.jpg",
        use_container_width=True
    )

    st.markdown("---")

    # 📊 Info Section
    st.markdown("## 📊 About Dashboard")
    st.markdown("""
    - ⚽ Football Analytics  
    - 🏏 Cricket Analytics  
    - 🤖 ML Predictions  
    - 📈 Player Insights  
    """)

    st.markdown("---")

    # 👤 Footer
    st.markdown("## 👨‍💻 Developer")
    st.markdown("Built with ❤️ using Streamlit")

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
.title {
    text-align:center;
    font-size:34px;
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
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #1f77b4;
    margin-bottom: 10px;
}

.sub-title {
    text-align: left;
    font-size: 30px;
    font-weight: 700;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚽🏏 Multi-Sport AI Analytics Dashboard</div>', unsafe_allow_html=True)

# ======================
# FOOTBALL
# ======================
if sport == "Football (EPL)":

    df = pd.read_csv("data/football_players.csv")

    df.rename(columns={
        "goals_scored": "goals",
        "ict_index": "ict",
        "clean_sheets": "clean",
        "goals_conceded": "conceded"
    }, inplace=True)

    st.header("⚽ Football Analytics")

    player = st.selectbox("Select Player", df["name"])
    row = df[df["name"] == player].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Goals", int(row["goals"]))
    col2.metric("Assists", int(row["assists"]))
    col3.metric("Influence", round(row["influence"],1))

    if st.button("Predict Score"):
        features = np.array([[row["goals"], row["assists"], row["minutes"],
                              row["influence"], row["creativity"], row["threat"],
                              row["ict"], row["form"], row["bps"],
                              row["clean"], row["conceded"]]])
        pred = model_f.predict(features)[0]
        st.success(f"🎯 Predicted Score: {round(pred,2)}")

    avg = df.mean(numeric_only=True)

    st.subheader("📊 Player vs Average")
    comp_df = pd.DataFrame({
        "Player": [row["goals"], row["assists"], row["influence"], row["creativity"], row["threat"]],
        "Average": [
            avg["goals"], avg["assists"],
            avg["influence"], avg["creativity"], avg["threat"]
        ]
    }, index=["Goals","Assists","Influence","Creativity","Threat"])

    st.bar_chart(comp_df)

    st.subheader("📈 Performance Trend")
    trend_df = pd.DataFrame({
        "Feature": ["Goals","Assists","Influence","Creativity","Threat"],
        "Player": [row["goals"], row["assists"], row["influence"], row["creativity"], row["threat"]],
        "Average": [
            avg["goals"], avg["assists"],
            avg["influence"], avg["creativity"], avg["threat"]
        ]
    }).set_index("Feature")

    st.line_chart(trend_df)

    st.subheader("🏆 Top Players")
    st.dataframe(df.sort_values("goals", ascending=False).head(10))

    st.subheader("📊 Goals Distribution")
    st.bar_chart(df["goals"].value_counts())

    st.subheader("📊 Ranking")
    pct = (df["goals"] < row["goals"]).mean() * 100
    rank = df["goals"].rank(ascending=False, method="min")
    player_rank = int(rank[df["name"] == player].values[0])

    col1, col2 = st.columns(2)
    col1.metric("Percentile", f"{round(pct,1)}%")
    col2.metric("Rank", f"{player_rank}/{len(df)}")

    st.markdown("---")
    st.subheader("⚔️ Player Comparison")

    p1_name = st.selectbox("Player 1", df["name"], key="f1")
    p2_name = st.selectbox("Player 2", df["name"], key="f2")

    p1 = df[df["name"] == p1_name].iloc[0]
    p2 = df[df["name"] == p2_name].iloc[0]

    comp = pd.DataFrame({
        p1_name: [p1["goals"], p1["assists"], p1["influence"]],
        p2_name: [p2["goals"], p2["assists"], p2["influence"]]
    }, index=["Goals","Assists","Influence"])

    st.line_chart(comp)

# ======================
# CRICKET
# ======================
elif sport == "Cricket (IPL)":

    df = pd.read_csv("data/player_stats.csv")

    st.header("🏏 Cricket Analytics")

    player = st.selectbox("Select Player", df["batsman"])
    row = df[df["batsman"] == player].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Runs", int(row["batsman_runs"]))
    col2.metric("Strike Rate", round(row["strike_rate"],1))
    col3.metric("Wickets", int(row["wickets"]))

    if st.button("Predict Impact"):
        features = np.array([[row["batsman_runs"], row["strike_rate"],
                              row["wickets"], row["is_four"], row["is_six"]]])
        pred = model_c.predict(features)[0]
        st.success(f"🏏 Impact Score: {round(pred,2)}")

    avg = df.mean(numeric_only=True)

    st.subheader("📊 Player vs Average")
    comp_df = pd.DataFrame({
        "Player": [row["batsman_runs"], row["strike_rate"], row["wickets"]],
        "Average": [
            avg["batsman_runs"], avg["strike_rate"], avg["wickets"]
        ]
    }, index=["Runs","Strike Rate","Wickets"])

    st.bar_chart(comp_df)

    st.subheader("📈 Performance Trend")
    trend_df = pd.DataFrame({
        "Feature": ["Runs","Strike Rate","Wickets","Fours","Sixes"],
        "Player": [row["batsman_runs"], row["strike_rate"], row["wickets"], row["is_four"], row["is_six"]],
        "Average": [
            avg["batsman_runs"], avg["strike_rate"],
            avg["wickets"], avg["is_four"], avg["is_six"]
        ]
    }).set_index("Feature")

    st.line_chart(trend_df)

    st.subheader("🏆 Top Players")
    st.dataframe(df.sort_values("batsman_runs", ascending=False).head(10))

    st.subheader("📊 Runs Distribution")
    st.bar_chart(df["batsman_runs"].value_counts())

    st.subheader("📊 Ranking")
    pct = (df["batsman_runs"] < row["batsman_runs"]).mean() * 100
    rank = df["batsman_runs"].rank(ascending=False, method="min")
    player_rank = int(rank[df["batsman"] == player].values[0])

    col1, col2 = st.columns(2)
    col1.metric("Percentile", f"{round(pct,1)}%")
    col2.metric("Rank", f"{player_rank}/{len(df)}")

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
