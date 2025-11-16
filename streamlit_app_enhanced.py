import streamlit as st
import os
import pandas as pd
from AIapi import EVAIAssistant
from model_utils import load_model, predict_range

# Streamlit page settings
st.set_page_config(page_title="EV Range Advisor", layout="wide")

# Simple minimal CSS
st.markdown("""
<style>
body { background: linear-gradient(180deg,#f2f7ff,#ffffff); }
.header { font-size:28px; font-weight:700; color:#0b3d91; margin-bottom:8px; }
.result-box { background:#e8ffe8; padding:10px; border-radius:8px; border-left:4px solid #34a853; }
.small { font-size:13px; color:#555; }
</style>
""", unsafe_allow_html=True)

# Load assistant + model
assistant = EVAIAssistant()
model = load_model()

# Sidebar
with st.sidebar:
    st.markdown("<div class='header'>EV Advisor</div>", unsafe_allow_html=True)
    if model is None:
        st.error("Model not found. Add ev_range_model.joblib.")
    else:
        st.success("Model loaded ✓")
    st.markdown("---")

st.markdown("<div class='header'>EV Range Advisor — Week 3</div>", unsafe_allow_html=True)

# For chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

# Layout
col1, col2 = st.columns([1,1.4])

with col1:

    st.subheader("Vehicle Inputs")

    battery = st.number_input("Battery capacity (kWh)", value=50.0, step=1.0)
    eff = st.number_input("Efficiency (km/kWh)", value=5.0, step=0.1)
    msrp = st.number_input("MSRP ($)", value=30000, step=100)
    daily_km = st.number_input("Daily driving (km)", value=30.0)

    make = st.text_input("Make", value="Generic")
    model_name = st.text_input("Model Name", value="EV")

    if st.button("Predict Range"):

        try:
            pred_km = predict_range(battery, eff)
            st.session_state["last_prediction"] = pred_km

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.write(f"**Predicted Range:** {pred_km:.2f} km")
            st.write(f"**Days Between Charges:** {pred_km/daily_km:.1f} days")
            st.write(f"**Cost per km (approx):** ${msrp/(pred_km*10):.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("AI Insights"):
                rec = assistant.recommend_vehicle(
                    {"make": make, "model": model_name, "battery_kwh": battery, "eff": eff},
                    pred_km
                )
                st.write("### Recommendation")
                st.write(rec)

                tips = assistant.maintenance_tips({"battery": battery})
                st.write("### Maintenance Tips")
                st.write(tips)

                charge = assistant.charging_strategy(daily_km, pred_km)
                st.write("### Charging Strategy")
                st.write(charge)

        except Exception as e:
            st.error(f"Prediction error: {e}")

with col2:
    st.subheader("AI Chat Assistant")

    user_msg = st.text_input("Ask anything about EVs")

    if st.button("Send"):
        if user_msg.strip() == "":
            st.warning("Enter a message.")
        else:
            context = ""
            if st.session_state["last_prediction"]:
                context = f"Predicted range: {st.session_state['last_prediction']} km. "

            prompt = context + user_msg

            try:
                reply = assistant.generate_text(prompt)
            except Exception:
                reply = assistant._fallback(prompt)

            st.session_state["chat_history"].insert(0, {"user": user_msg, "assistant": reply})
            st.session_state["chat_input"] = ""

    # Display chat
    for chat in st.session_state["chat_history"]:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Assistant:** {chat['assistant']}")
        st.markdown("---")

st.markdown("<hr><div class='small'>EV Advisor — Built for Week 3 Project</div>", unsafe_allow_html=True)
