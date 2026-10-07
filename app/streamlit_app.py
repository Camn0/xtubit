"""X-TUBIT Interactive Dashboard.

Provides interactive visualization of:
  1. High-throughput Candidate Ranking & Drug-likeness
  2. Bayesian Epistemic Uncertainty & Multi-Objective Pareto Frontier
  3. Quantum/Digital Annealing Solver Benchmark (Exact, TApSA, SpSA, tSB)
  4. Pks13-TE Docking RMSD & Conformation Validation
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="X-TUBIT | Pks13-TE Screening Platform", page_icon="🧬")

st.title("X-TUBIT: High-Throughput In Silico Screening Platform")
st.markdown(
    "**Bayesian Invariant Neural Networks & Digital Annealing Solvers for Pks13-TE Inhibitor Discovery**  \n"
    "*Faculty of Engineering, Universitas Indonesia*"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Candidate Screening & Ranking",
    "📈 Bayesian Uncertainty & Pareto Front",
    "⚡ Digital Annealing Solvers",
    "🔬 3D Docking & Validation"
])

selected_file = Path("data/processed/selected.parquet")
metrics_file = Path("data/processed/metrics/summary.json")
solver_file = Path("data/processed/solver_out/solver_comparison.parquet")

# Tab 1: Candidate Ranking
with tab1:
    st.subheader("Selected Candidates (Multi-Objective qPMHI)")
    if selected_file.exists():
        df_sel = pd.read_parquet(selected_file)
        cols_to_show = ["rank", "mol_id", "smiles_can", "qpmhi_score", "mu", "sigma", "qed", "sa", "mw", "logp"]
        display_cols = [c for c in cols_to_show if c in df_sel.columns]
        st.dataframe(df_sel[display_cols], use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Filtered Candidates", len(df_sel))
        col_b.metric("Top Candidate", df_sel.iloc[0]["mol_id"])
        col_c.metric("Top qPMHI Acquisition Score", f"{df_sel.iloc[0]['qpmhi_score']:.4f}")
    else:
        st.info("Artifact `data/processed/selected.parquet` not found. Run the execution pipeline first (`python scripts/run_pipeline.py`).")

# Tab 2: Bayesian Uncertainty & Pareto Front
with tab2:
    st.subheader("Bayesian Epistemic Uncertainty Calibration")
    if selected_file.exists():
        df_sel = pd.read_parquet(selected_file)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Affinity Prediction vs. Epistemic Uncertainty")
            st.scatter_chart(df_sel, x="mu", y="sigma", color="mol_id")
        with c2:
            st.markdown("#### Multi-Objective Pareto Frontier: Affinity vs. QED")
            st.scatter_chart(df_sel, x="mu", y="qed", color="mol_id")
    else:
        st.info("Execute pipeline to generate Bayesian surrogate predictions.")

# Tab 3: Digital Annealing Solver Comparison
with tab3:
    st.subheader("Hamiltonian Minimization: Digital Annealing Benchmarks")
    if solver_file.exists():
        df_solv = pd.read_parquet(solver_file)
        st.dataframe(df_solv, use_container_width=True)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("#### Minimum Ground-State Energy Reached")
            st.bar_chart(df_solv.set_index("solver")["energy"])
        with s_col2:
            st.markdown("#### Time-to-Solution (TTS99 in seconds)")
            st.bar_chart(df_solv.set_index("solver")["tts_99"])
    else:
        st.info("Execute pipeline to generate solver comparison metrics.")

# Tab 4: 3D Docking & Validation
with tab4:
    st.subheader("Pks13-TE Binding Validation & Heavy-Atom RMSD")
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Target Crystal Structure", metrics_data.get("reference_pdb", "5V3Y"))
        m2.metric("Lead Compound", metrics_data.get("lead_compound", "TAM16"))
        rmsd = metrics_data.get("heavy_atom_rmsd_A", 0.0)
        m3.metric("Heavy-Atom RMSD", f"{rmsd:.2f} Å", delta="Pass (< 2.0 Å)" if rmsd < 2.0 else "Fail")
        m4.metric("Constraint Violations", metrics_data.get("constraint_violations", 0))

        st.json(metrics_data)
    else:
        st.info("Execute pipeline to generate validation and RMSD metrics.")

st.sidebar.markdown("### Execution Command")
st.sidebar.code("python scripts/run_pipeline.py", language="bash")
st.sidebar.markdown("### Interactive Dashboard")
st.sidebar.code("streamlit run app/streamlit_app.py", language="bash")
st.sidebar.markdown("---")
st.sidebar.caption("Human-in-the-Loop records append immutable JSONL entries without mutating upstream artifacts.")
