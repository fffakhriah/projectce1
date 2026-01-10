import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from aco_single import ACO_Single
from aco_multi import ACO_Multi

# =========================
# PAGE SETUP
# =========================
st.set_page_config(layout="wide")
st.title("🎬 Cinema Ticket Pricing Optimization using Ant Colony Optimization")

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("cinema_ticket_pricing_clean.csv")
data.columns = data.columns.str.lower().str.replace(" ", "_")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Optimization Settings")

    mode = st.selectbox(
        "Optimization Mode",
        ["Single Objective", "Multi Objective"]
    )

    ants = st.slider("Number of Persons", 10, 50, 20)
    iterations = st.slider("Iterations", 20, 100, 50)
    evap = st.slider("Evaporation Rate", 0.1, 0.9, 0.5)

    if mode == "Multi Objective":
        st.subheader("Objective Weights")
        w1 = st.slider("Revenue Weight", 0.0, 1.0, 0.6)
        w2 = st.slider("Demand Weight", 0.0, 1.0, 0.3)
        w3 = st.slider("Price Penalty", 0.0, 1.0, 0.1)

    run_btn = st.button("🚀 Run Optimization")

# =========================
# MAIN RESULT AREA
# =========================
if run_btn:
    if mode == "Single Objective":
        model = ACO_Single(data, ants, iterations, evap)
        price, fitness, history = model.run()
        st.subheader("📈 Single Objective Optimization Result")
    else:
        model = ACO_Multi(data, ants, iterations, w1, w2, w3, evap)
        price, fitness, history = model.run()
        st.subheader("📊 Multi Objective Optimization Result")

    # Results text
    st.markdown(f"""
    **Optimal Ticket Price:** RM {price:.2f}  
    **Fitness Value:** {fitness:.2f}
    """)

    # Convergence Curve 
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness Value")
    ax.set_title("ACO Convergence Curve")

    st.pyplot(fig)
