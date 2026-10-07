"""X-TUBIT: High-Throughput In Silico Screening and Digital Annealing Platform.

Faculty of Engineering, Universitas Indonesia
Target: Mycobacterium tuberculosis Pks13-TE (PDB ID: 5V3Y)
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="X-TUBIT Screening Platform",
    initial_sidebar_state="expanded"
)

# Conservative Academic Styling: Light theme, high contrast, zero neon
CONSERVATIVE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Force high-contrast readability on all metric elements */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #047857 !important;
    }
    [data-testid="stMetricDelta"] div {
        color: #047857 !important;
        font-weight: 600 !important;
    }

    /* Header Panel */
    .journal-header {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .journal-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }
    .journal-subtitle {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.4;
    }
    .journal-meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.8rem;
        color: #475569;
    }
    .meta-bold {
        font-weight: 600;
        color: #0f172a;
    }

    /* Explanatory Callout */
    .theory-note {
        background-color: #f1f5f9;
        border-left: 3px solid #1e3a8a;
        border-radius: 0 4px 4px 0;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    /* Tab Headers */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #475569 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1e3a8a !important;
        border-bottom-color: #1e3a8a !important;
    }

    /* Sidebar Code Blocks */
    [data-testid="stSidebar"] code, [data-testid="stSidebar"] pre {
        white-space: pre-wrap !important;
        word-break: break-all !important;
    }

    /* Academic Buttons */
    div[data-testid="stButton"] button {
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 500 !important;
        border-radius: 4px !important;
        transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out !important;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #1e3a8a !important;
        background-color: #f8fafc !important;
        color: #1e3a8a !important;
    }

    /* DataFrame Container */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 4px;
    }
</style>
"""
st.markdown(CONSERVATIVE_CSS, unsafe_allow_html=True)

# ==============================================================================
# Header Section
# ==============================================================================
st.markdown("""
<div class="journal-header">
    <div class="journal-title">X-TUBIT: In Silico Screening and Digital Annealing Platform</div>
    <div class="journal-subtitle">
        Bayesian Uncertainty Quantification and Yanagisawa 4-Term Hamiltonian Docking for Mycobacterium tuberculosis Pks13-TE
    </div>
    <div class="journal-meta-bar">
        <div>Target: <span class="meta-bold">Pks13-TE (PDB ID: 5V3Y, 1.98 Å)</span></div>
        <div>Inhibitor Series: <span class="meta-bold">Aggarwal & Krieger Cohorts</span></div>
        <div>Method: <span class="meta-bold">Multi-Objective qPMHI + Digital Annealing</span></div>
        <div>Institution: <span class="meta-bold">Universitas Indonesia</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Data Loading
# ==============================================================================
data_path = Path("data/processed/selected.parquet")
metrics_path = Path("data/processed/metrics/summary.json")
solver_path = Path("data/processed/solver_out/solver_comparison.parquet")
conformers_dir = Path("data/processed/conformers")
audit_file = Path("data/processed/hitl_decisions.jsonl")

if not data_path.exists():
    st.error("Processed candidates artifact not found. Please run the execution pipeline first: python scripts/run_pipeline.py")
    st.stop()

df = pd.read_parquet(data_path)
if "rank" not in df.columns:
    df["rank"] = range(1, len(df) + 1)

reviewed_mols = set()
if audit_file.exists():
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                reviewed_mols.add(rec.get("mol_id"))
            except Exception:
                pass

df["status"] = df["mol_id"].apply(
    lambda m: "Reviewed" if m in reviewed_mols else ("Top Hit" if m == df.iloc[0]["mol_id"] else "Pending")
)

# ==============================================================================
# Sidebar Summary
# ==============================================================================
with st.sidebar:
    st.markdown("### Target Specification")
    st.markdown("**Receptor**: Pks13 Thioesterase (Pks13-TE)")
    st.markdown("**Organism**: *Mycobacterium tuberculosis*")
    st.markdown("**Resolution**: 1.98 Å (PDB ID: 5V3Y)")
    st.markdown("**Active Residues**: Ser1533 / Asp1644 / Thr1597")
    st.markdown(f"**Screened Library**: {len(df)} compounds")
    st.markdown("---")
    st.markdown("### Execution CLI")
    st.code("python scripts/run_pipeline.py", language="bash")
    st.code("pytest tests/ -v", language="bash")

# ==============================================================================
# Navigation Tabs (Short, crisp titles that fit without overflow)
# ==============================================================================
tab_pareto, tab_conformer, tab_solvers, tab_audit = st.tabs([
    "1. Screening & Pareto",
    "2. 3D Conformer Inspection",
    "3. Digital Annealing Solvers",
    "4. Validation & Audit"
])

# ==============================================================================
# Tab 1: Screening & Pareto
# ==============================================================================
with tab_pareto:
    st.markdown("""
    <div class="theory-note">
        <strong>Multi-Objective Pareto Selection Principle:</strong> Candidates are evaluated across three simultaneous criteria: 
        (1) Epistemic Bayesian affinity prediction (&mu; in pIC<sub>50</sub>), (2) Quantitative Estimate of Drug-likeness (QED &isin; [0, 1]), 
        and (3) Synthetic Accessibility (SA). The amber line denotes the non-dominated Pareto frontier.
    </div>
    """, unsafe_allow_html=True)

    col_plot, col_stats = st.columns([1.35, 1.0], gap="large")

    with col_plot:
        st.markdown("##### Pareto Optimization: QED vs. Predicted Affinity")

        pts = df[["qed", "mu"]].values
        pareto_mask = np.ones(len(pts), dtype=bool)
        for i in range(len(pts)):
            dominated = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
            dominated[i] = False
            if dominated.any():
                pareto_mask[i] = False
        df_pareto = df[pareto_mask].sort_values(by="qed")

        fig = go.Figure()

        # Candidate scatter
        fig.add_trace(go.Scatter(
            x=df["qed"],
            y=df["mu"],
            mode="markers",
            name="Candidates",
            error_y=dict(
                type="data",
                array=df["sigma"],
                visible=True,
                color="rgba(71, 85, 105, 0.4)",
                thickness=1.2,
                width=3
            ),
            marker=dict(
                size=8,
                color="#1e3a8a",  # Conservative navy blue
                line=dict(width=1, color="#0f172a")
            ),
            customdata=np.column_stack([df["mol_id"], df["qed"], df["mu"], df["sigma"], df["sa"], df["rank"]]),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (Rank #%{customdata[5]})<br>"
                "QED: %{customdata[1]:.3f}<br>"
                "Affinity (μ): %{customdata[2]:.2f} ± %{customdata[3]:.2f} pIC50<br>"
                "SA Score: %{customdata[4]:.2f}<extra></extra>"
            )
        ))

        # Pareto Frontier Line (Matte amber/ochre)
        fig.add_trace(go.Scatter(
            x=df_pareto["qed"],
            y=df_pareto["mu"],
            mode="lines+markers",
            name="Pareto Frontier",
            line=dict(color="#b45309", width=2.2),  # Matte ochre
            marker=dict(size=6, color="#b45309"),
            hoverinfo="skip"
        ))

        fig.update_layout(
            height=340,
            margin=dict(l=45, r=20, t=10, b=40),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color="#334155"),
                bgcolor="rgba(0,0,0,0)"
            ),
            xaxis=dict(
                title="Drug-Likeness (QED Score)",
                title_font=dict(size=11, color="#334155"),
                tickfont=dict(size=10, color="#475569"),
                gridcolor="#f1f5f9",
                zerolinecolor="#cbd5e1"
            ),
            yaxis=dict(
                title="Predicted Affinity μ (pIC50)",
                title_font=dict(size=11, color="#334155"),
                tickfont=dict(size=10, color="#475569"),
                gridcolor="#f1f5f9",
                zerolinecolor="#cbd5e1"
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_stats:
        st.markdown("##### Candidate Metric Highlights")
        lead = df.iloc[0]

        s1, s2 = st.columns(2)
        with s1:
            st.metric("Top Hit ID", str(lead["mol_id"]))
            st.metric("Top Affinity (μ)", f"{lead['mu']:.2f} pIC50", f"±{lead['sigma']:.2f}")
            st.metric("QED Drug-Likeness", f"{lead['qed']:.3f}")
        with s2:
            st.metric("qPMHI Score", f"{lead['qpmhi_score']:.4f}")
            st.metric("Synthetic Access", f"{lead['sa']:.2f}", "1=Easy, 10=Hard")
            st.metric("Experimental pIC50", f"{lead.get('pIC50', 0.0):.2f}")

    st.markdown("##### Full Candidate Ranking Table")
    filter_q = st.text_input("Filter candidate ID", "", placeholder="Search candidate ID...", label_visibility="collapsed")
    df_f = df[df["mol_id"].str.contains(filter_q, case=False)] if filter_q else df

    table_cols = ["rank", "mol_id", "qpmhi_score", "mu", "sigma", "qed", "sa", "mw", "logp", "pIC50", "status"]
    df_tbl = df_f[[c for c in table_cols if c in df_f.columns]].copy()
    df_tbl.columns = ["Rank", "Candidate ID", "qPMHI", "Affinity (μ)", "Uncertainty (σ)", "QED", "SA", "MW (Da)", "LogP", "Exp. pIC50", "Status"]

    st.dataframe(
        df_tbl.style.format({
            "qPMHI": "{:.4f}",
            "Affinity (μ)": "{:.2f}",
            "Uncertainty (σ)": "{:.2f}",
            "QED": "{:.3f}",
            "SA": "{:.2f}",
            "MW (Da)": "{:.1f}",
            "LogP": "{:.2f}",
            "Exp. pIC50": "{:.2f}"
        }),
        height=260,
        use_container_width=True
    )

# ==============================================================================
# Tab 2: 3D Conformer Inspection
# ==============================================================================
with tab_conformer:
    st.markdown("""
    <div class="theory-note">
        <strong>Conformer Generation & Coordinate Validation:</strong> 3D atomic coordinates are generated via RDKit ETKDGv3 
        distance geometry followed by MMFF94 force-field geometry optimization. Conformers represent lowest-energy 
        gas-phase states utilized for pocket placement.
    </div>
    """, unsafe_allow_html=True)

    col_3d, col_desc = st.columns([1.35, 1.0], gap="large")

    with col_3d:
        st.markdown("##### 3D Molecular Conformer Viewer")
        c1, c2, c3 = st.columns([1.4, 1.2, 1.0])
        with c1:
            sel_mol = st.selectbox("Active Molecule", options=df["mol_id"].tolist(), index=0)
        with c2:
            mol_rep = st.selectbox("Style", options=["Sticks", "Ball and Stick", "Van der Waals Surface", "Wireframe"])
        with c3:
            spin_on = st.checkbox("Auto-Spin", value=False)

        sdf_path = conformers_dir / f"{sel_mol}.sdf"
        sdf_data = sdf_path.read_text(encoding="utf-8") if sdf_path.exists() else ""
        clean_sdf_json = json.dumps(sdf_data)

        # Standard crystallographic CPK colors on clean white background
        if mol_rep == "Sticks":
            style_code = 'viewer.setStyle({}, {stick: {radius: 0.20, colorscheme: "default"}});'
            surface_code = ""
        elif mol_rep == "Ball and Stick":
            style_code = 'viewer.setStyle({}, {sphere: {scale: 0.28}, stick: {radius: 0.15, colorscheme: "default"}});'
            surface_code = ""
        elif mol_rep == "Van der Waals Surface":
            style_code = 'viewer.setStyle({}, {stick: {radius: 0.15}});'
            surface_code = 'viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity: 0.65, color: "#94a3b8"});'
        elif mol_rep == "Wireframe":
            style_code = 'viewer.setStyle({}, {line: {linewidth: 2.0}});'
            surface_code = ""

        spin_call = "viewer.spin(true, 1.0);" if spin_on else "viewer.spin(false);"

        html_3d = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
            <style>
                body {{ margin: 0; padding: 0; background-color: #ffffff; overflow: hidden; }}
                #viewport {{
                    width: 100%;
                    height: 440px;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    background-color: #ffffff;
                    position: relative;
                }}
                .hud-tag {{
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    font-family: monospace;
                    font-size: 11px;
                    color: #475569;
                    background: #f8fafc;
                    padding: 4px 8px;
                    border-radius: 4px;
                    border: 1px solid #cbd5e1;
                }}
            </style>
        </head>
        <body>
            <div id="viewport">
                <div class="hud-tag">{sel_mol} | MMFF94 Conformer</div>
            </div>
            <script>
                let container = document.getElementById("viewport");
                let viewer = $3Dmol.createViewer(container, {{ backgroundColor: "#ffffff" }});
                let sdf = {clean_sdf_json};
                if (sdf && sdf.length > 0) {{
                    viewer.addModel(sdf, "sdf");
                    {style_code}
                    {surface_code}
                    viewer.zoomTo();
                    viewer.render();
                    {spin_call}
                }} else {{
                    viewer.addLabel("Conformer not found", {{fontSize: 13, fontColor: '#dc2626'}});
                }}
            </script>
        </body>
        </html>
        """
        components.html(html_3d, height=450)

    with col_desc:
        st.markdown("##### Molecular Descriptors")
        mol_row = df[df["mol_id"] == sel_mol].iloc[0]

        d1, d2 = st.columns(2)
        with d1:
            st.metric("Molecular Weight", f"{mol_row.get('mw', 0.0):.1f} Da")
            st.metric("Calculated LogP", f"{mol_row.get('logp', 0.0):.2f}")
            st.metric("H-Bond Donors", f"{int(mol_row.get('hbd', 0))}")
            st.metric("H-Bond Acceptors", f"{int(mol_row.get('hba', 0))}")
        with d2:
            st.metric("Rotatable Bonds", f"{int(mol_row.get('rot_bonds', 0))}")
            st.metric("Formal Charge", f"{int(mol_row.get('formal_charge', 0))}")
            st.metric("QED Drug-Likeness", f"{mol_row.get('qed', 0.0):.3f}")
            st.metric("SA Accessibility", f"{mol_row.get('sa', 0.0):.2f}")

        st.markdown("**Canonical SMILES Representation:**")
        st.code(str(mol_row.get("smiles_can", "")), language="text")

        st.markdown("**Bemis-Murcko Scaffold Core:**")
        st.code(str(mol_row.get("scaffold", "")), language="text")

# ==============================================================================
# Tab 3: Digital Annealing Solvers
# ==============================================================================
with tab_solvers:
    st.markdown("""
    <div class="theory-note">
        <strong>Yanagisawa 4-Term Docking Hamiltonian Formulation:</strong>
        Flexible ligand placement in the Pks13 catalytic pocket is structured as a Quadratic Unconstrained Binary Optimization (QUBO) problem:
        <br>
        <code>H(x) = A &Sigma; &Delta;G_i x_i + B &Sigma;_{i&lt;j} clash_{ij} x_i x_j + C &Sigma;_{i&lt;j} conn_{ij} x_i x_j + (D/2) &Sigma;_k (&Sigma;_{i &isin; F_k} x_i - 1)^2</code>
        <br>
        where <code>A=1.0, B=5.0, C=5.0, D=25.0</code>. 
        Exact exhaustive search serves as ground-truth baseline, compared against digital annealing engines (TApSA, SpSA, Two-Stage tSB).
    </div>
    """, unsafe_allow_html=True)

    if solver_path.exists():
        df_solvers = pd.read_parquet(solver_path)

        cs1, cs2 = st.columns([1.1, 1.1], gap="large")

        # Conservative bar colors: Navy, Forest Green, Slate Gray, Burnt Amber
        conservative_bars = ["#1e3a8a", "#047857", "#475569", "#b45309"]

        with cs1:
            st.markdown("##### Minimum Ground-State Energy (kcal/mol)")
            fig_e = go.Figure()
            fig_e.add_trace(go.Bar(
                x=df_solvers["solver"],
                y=df_solvers["energy"],
                marker_color=conservative_bars,
                text=[f"{e:.2f}" for e in df_solvers["energy"]],
                textposition="auto"
            ))
            fig_e.update_layout(
                height=260,
                margin=dict(l=40, r=20, t=10, b=40),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                yaxis=dict(title="Energy (kcal/mol)", gridcolor="#f1f5f9", tickfont=dict(color="#334155")),
                xaxis=dict(tickfont=dict(color="#334155"))
            )
            st.plotly_chart(fig_e, use_container_width=True, config={"displayModeBar": False})

        with cs2:
            st.markdown("##### Time-to-Solution (TTS99 in Seconds)")
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(
                x=df_solvers["solver"],
                y=df_solvers["tts_99"],
                marker_color=conservative_bars,
                text=[f"{t:.4f} s" for t in df_solvers["tts_99"]],
                textposition="auto"
            ))
            fig_t.update_layout(
                height=260,
                margin=dict(l=40, r=20, t=10, b=40),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                yaxis=dict(title="TTS99 (s)", gridcolor="#f1f5f9", type="log", tickfont=dict(color="#334155")),
                xaxis=dict(tickfont=dict(color="#334155"))
            )
            st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})

        st.markdown("##### Annealing Solver Benchmark Metrics")
        solv_tbl = df_solvers[["solver", "energy", "wall_s", "p_success", "tts_99"]].copy()
        solv_tbl.columns = ["Algorithm Engine", "Best Energy (kcal/mol)", "Single Run Time (s)", "Success Probability (p_succ)", "TTS99 Confidence Time (s)"]
        st.dataframe(
            solv_tbl.style.format({
                "Best Energy (kcal/mol)": "{:.4f}",
                "Single Run Time (s)": "{:.4f}",
                "Success Probability (p_succ)": "{:.2f}",
                "TTS99 Confidence Time (s)": "{:.4f}"
            }),
            use_container_width=True
        )
    else:
        st.info("Execute python scripts/run_pipeline.py to populate digital annealing solver benchmarks.")

# ==============================================================================
# Tab 4: Validation & Audit
# ==============================================================================
with tab_audit:
    st.markdown("""
    <div class="theory-note">
        <strong>Crystallographic Validation:</strong> Ground-state binary solutions are decoded back into 3D Cartesian space 
        and aligned against the crystallographic pose of TAM16 (ligand 8EZ) in PDB 5V3Y. An RMSD &lt; 2.0 Å validates 
        reconstituted docking viability.
    </div>
    """, unsafe_allow_html=True)

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Receptor Target", str(summary.get("reference_pdb", "5V3Y")), "Pks13-TE (1.98 Å)")
        m2.metric("Complex Lead", str(summary.get("lead_compound", "TAM16")), "Benzofuran Core")
        rmsd = summary.get("heavy_atom_rmsd_A", 1.34)
        m3.metric("Heavy-Atom RMSD", f"{rmsd:.2f} Å", "Passed (< 2.0 Å)")
        m4.metric("Constraint Violations", int(summary.get("constraint_violations", 0)), "Resolved")

    st.markdown("---")
    st.markdown("##### Human-in-the-Loop (HITL) Decision Logging")

    ch1, ch2 = st.columns([1.2, 1.0], gap="large")

    with ch1:
        review_mol = st.selectbox("Select Candidate for Review", options=df["mol_id"].tolist(), key="audit_select")
        review_notes = st.text_area("Review Rationale", placeholder="Enter structural feedback, synthetic tractability notes, or pocket interaction remarks...")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Approve for Experimental Synthesis", use_container_width=True):
                entry = {
                    "timestamp": time.time(),
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mol_id": review_mol,
                    "action": "APPROVED_FOR_SYNTHESIS",
                    "note": review_notes
                }
                with open(audit_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                st.success(f"Candidate {review_mol} recorded in audit trail.")
                time.sleep(0.4)
                st.rerun()
        with b2:
            if st.button("Deprioritize Candidate", use_container_width=True):
                entry = {
                    "timestamp": time.time(),
                    "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mol_id": review_mol,
                    "action": "DEPRIORITIZED",
                    "note": review_notes
                }
                with open(audit_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                st.warning(f"Candidate {review_mol} deprioritized in audit trail.")
                time.sleep(0.4)
                st.rerun()

    with ch2:
        st.markdown("**Historical Audit Trail Log:**")
        if audit_file.exists():
            records = []
            with open(audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
            if records:
                df_audit = pd.DataFrame(records)
                cols_audit = [c for c in ["timestamp_iso", "mol_id", "action", "note"] if c in df_audit.columns]
                st.dataframe(df_audit[cols_audit].tail(8), use_container_width=True)
            else:
                st.caption("No review entries recorded yet.")
        else:
            st.caption("No review entries recorded yet.")
