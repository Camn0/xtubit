import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="X-TUBIT")
st.title("X-TUBIT · Pks13-TE Screening")

sel_path = Path("data/processed/selected.parquet")
if sel_path.exists():
    sel = pd.read_parquet(sel_path)
    st.subheader("Candidate ranking")
    st.dataframe(sel, use_container_width=True)
    if {"qed","mu"}.issubset(sel.columns):
        st.scatter_chart(sel, x="qed", y="mu")
else:
    st.info("No selected.parquet artifact yet. Run B1-B5 first.")

st.caption("HITL actions should append immutable JSONL records; they must not mutate upstream experiment artifacts.")
