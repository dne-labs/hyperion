import os
import httpx
import streamlit as st

st.set_page_config(page_title="Hyperion", layout="wide")
st.title("Hyperion — Inference OS Simulator")
url = os.getenv("HYPERION_GATEWAY_URL", "http://localhost:8000")
try:
    topology = httpx.get(f"{url}/topology", timeout=2).json()
    state = topology["state"]
    a, b, c = st.columns(3)
    a.metric("Prefill replicas", state["prefill_replicas"])
    b.metric("Decode replicas", state["decode_replicas"])
    c.metric("Prefix entries", topology["cache_entries"])
    st.subheader("Controller topology")
    st.json(topology)
    st.caption("Refresh after sending gateway traffic. This is CPU-only simulated inference; Vidur supplies offline capacity studies.")
except Exception as error:
    st.error(f"Gateway unavailable: {error}")
