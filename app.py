import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from aco_single import ACO_Single
from aco_multi import ACO_Multi

st.title("🎬 Cinema Ticket Pricing Optimization using ACO")

data = pd.read_csv("cinema_ticket_pricing_clean.csv")
data.columns = data.columns.str.lower().str.replace(" ", "_")

mode = st.selectbox("Optimization Mode", ["Single Objective", "Multi Objective"])

ants = st.slider("Number of Ants", 10, 50, 20)
iterations = st.slider("Iterations", 20, 100, 50)
evap = st.slider("Evaporation Rate", 0.1, 0.9, 0.5)

if mode == "Multi Objective":
    w1 = st.slider("Revenue Weight", 0.0, 1.0, 0.6)
    w2 = st.slider("Demand Weight", 0.0, 1.0, 0.3)
    w3 = st.slider("Price Penalty", 0.0, 1.0, 0.1)

if st.button("Run Optimization"):
    if mode == "Single Objective":
        model = ACO_Single(data, ants, iterations, evap)
    else:
        model = ACO_Multi(data, ants, iterations, w1, w2, w3, evap)

    price, fitness, history = model.run()

    st.success(f"Optimal Ticket Price: RM {price:.2f}")
    st.write("Fitness Value:", fitness)

    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)
