import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Multi-Sport AI Dashboard",
    layout="wide",
    page_icon="⚽"
)

# ======================
# LOAD MODELS
# ======================
model_f = pickle.load(open("models/football_model.pkl", "rb"))
model_c = pickle.load(open("models/cricket_model.pkl", "rb"))

# ======================
# HEADER
# ======================
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
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
def kpi_card(title, value):
    st.markdown(f"""
        <div style="
            padding:15px;
            border-radius:10px;
            background-color:#1E1E1E;
            text-align:center;
            color:white;">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
    """, unsafe_allow_html=True)

# ======================
# ⚽ FOOTBALL DASHBOARD
# ======================
if sport == "Football (EPL)":

    st.subheader("⚽ Football Player Performance")

    col1, col2, col3, col4 = st.columns(4)

    goals = col1.number_input("Goals", 0, 50)
    assists = col2.number_input("Assists", 0, 30)
    minutes = col3.number_input("Minutes", 0, 4000)
    form = col4.number_input("Form", 0.0, 20.0)

    col5, col6, col7, col8 = st.columns(4)

    influence = col5.number_input("Influence", 0.0, 1000.0)
    creativity = col6.number_input("Creativity", 0.0, 1000.0)
    threat = col7.number_input("Threat", 0.0, 1000.0)
    ict = col8.number_input("ICT Index", 0.0, 1000.0)

    col9, col10 = st.columns(2)

    bps = col9.number_input("BPS", 0, 1000)
    clean = col10.number_input("Clean Sheets", 0, 50)
    conceded = st.number_input("Goals Conceded", 0, 100)

    if st.button("🔮 Predict Football Score"):

        features = np.array([[goals, assists, minutes, influence, creativity,
                              threat, ict, form, bps, clean, conceded]])

        prediction = model_f.predict(features)[0]

        st.markdown("### 📊 Prediction Result")

        k1, k2, k3 = st.columns(3)
        k1.metric("Predicted Score", round(prediction, 2))
        k2.metric("Goals", goals)
        k3.metric("Assists", assists)

        st.markdown("---")

        st.subheader("📈 Feature Importance")

        df_feat = pd.DataFrame({
            "Feature": [
                "Goals", "Assists", "Minutes", "Influence",
                "Creativity", "Threat", "ICT", "Form",
                "BPS", "Clean Sheets", "Conceded"
            ],
            "Importance": model_f.coef_
        })

        st.bar_chart(df_feat.set_index("Feature"))

# ======================
# 🏏 CRICKET DASHBOARD
# ======================
elif sport == "Cricket (IPL)":

    st.subheader("🏏 Cricket Player Performance")

    col1, col2, col3 = st.columns(3)

    runs = col1.number_input("Runs", 0, 5000)
    strike_rate = col2.number_input("Strike Rate", 0.0, 300.0)
    wickets = col3.number_input("Wickets", 0, 500)

    col4, col5 = st.columns(2)

    fours = col4.number_input("Fours", 0, 500)
    sixes = col5.number_input("Sixes", 0, 300)

    if st.button("🔮 Predict Cricket Impact"):

        features = np.array([[runs, strike_rate, wickets, fours, sixes]])

        prediction = model_c.predict(features)[0]

        st.markdown("### 📊 Prediction Result")

        k1, k2, k3 = st.columns(3)
        k1.metric("Impact Score", round(prediction, 2))
        k2.metric("Runs", runs)
        k3.metric("Wickets", wickets)

        st.markdown("---")

        st.subheader("📈 Feature Importance")

        df_feat = pd.DataFrame({
            "Feature": ["Runs", "Strike Rate", "Wickets", "Fours", "Sixes"],
            "Importance": model_c.coef_
        })

        st.bar_chart(df_feat.set_index("Feature"))

# ======================
# 📂 DATA EXPLORER
# ======================
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Dataset")

file = st.sidebar.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Correlation Matrix")
    st.dataframe(df.corr())
