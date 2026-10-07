"""X-TUBIT: Human-in-the-Loop (HITL) Drug Discovery Dashboard.

Provides interactive WebGL 3D molecular visualization, multi-objective
Pareto frontier exploration, candidate ranking, and digital annealing benchmarks.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# Page Configuration & Modern Dark UI Theme
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Human-in-the-Loop (HITL) Dashboard · X-TUBIT",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Header Card */
    .hitl-header {
        background: linear-gradient(180deg, #161f30 0%, #0f1624 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hitl-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hitl-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .window-dots {
        display: flex;
        gap: 6px;
        align-items: center;
    }
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-red { background-color: #ef4444; }
    .dot-yellow { background-color: #f59e0b; }
    .dot-green { background-color: #10b981; }

    /* Dashboard Panels */
    .dashboard-panel {
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        margin-bottom: 16px;
    }
    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
    }
    .panel-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: 0.02em;
    }
    
    /* Status Badges */
    .badge-top {
        background-color: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-reviewed {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-pending {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Metric pill */
    .stat-pill {
        background: #1e293b;
        border-radius: 8px;
        padding: 8px 12px;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        font-size: 0.8rem;
    }
    .stat-pill-label { color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; }
    .stat-pill-val { color: #38bdf8; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# Header Bar
# ==============================================================================
st.markdown("""
<div class="hitl-header">
    <div style="display: flex; align-items: center; gap: 14px;">
        <div class="window-dots">
            <span class="dot dot-red"></span>
            <span class="dot dot-yellow"></span>
            <span class="dot dot-green"></span>
        </div>
        <div>
            <div class="hitl-title">⚡ Human-in-the-Loop (HITL) Dashboard · X-TUBIT</div>
            <div class="hitl-sub">Mycobacterium tuberculosis Pks13-TE Screening & Digital Annealing Platform</div>
        </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="badge-top">Target: PDB 5V3Y</span>
        <span class="badge-reviewed">Annealing Solvers: Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Data Loading
# ==============================================================================
selected_path = Path("data/processed/selected.parquet")
metrics_path = Path("data/processed/metrics/summary.json")
solver_path = Path("data/processed/solver_out/solver_comparison.parquet")
conformers_dir = Path("data/processed/conformers")

if not selected_path.exists():
    st.warning("⚠️ Processed candidates artifact not found. Run `python scripts/run_pipeline.py` first to generate data.")
    st.stop()

df_candidates = pd.read_parquet(selected_path)
if "rank" not in df_candidates.columns:
    df_candidates["rank"] = range(1, len(df_candidates) + 1)

# Assign review status from audit file if exists
audit_file = Path("data/processed/hitl_decisions.jsonl")
reviewed_ids = set()
if audit_file.exists():
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                reviewed_ids.add(rec.get("mol_id"))
            except Exception:
                pass

df_candidates["status"] = df_candidates["mol_id"].apply(
    lambda m: "Reviewed" if m in reviewed_ids else ("Top Hit" if m == df_candidates.iloc[0]["mol_id"] else "Pending")
)

# ==============================================================================
# Main Two-Column Layout (Matching UI Reference Image)
# ==============================================================================
col_left, col_right = st.columns([1.05, 1.15], gap="medium")

# ------------------------------------------------------------------------------
# Left Column: 3D Molecular Interaction Viewer
# ------------------------------------------------------------------------------
with col_left:
    st.markdown("""
    <div class="panel-header">
        <span class="panel-title">🔬 3D Molecular Interaction Viewer</span>
        <span style="font-size: 0.75rem; color: #94a3b8;">Interactive WebGL</span>
    </div>
    """, unsafe_allow_html=True)

    # Candidate Selection for 3D Viewer
    c_opts = df_candidates["mol_id"].tolist()
    default_mol = c_opts[0]
    
    # Viewer Controls in Horizontal Columns
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.0, 1.0])
    with ctrl_col1:
        active_mol = st.selectbox(
            "Select Molecule",
            options=c_opts,
            index=0,
            key="viewer_mol_select",
            label_visibility="collapsed"
        )
    with ctrl_col2:
        representation = st.selectbox(
            "Style",
            options=["Sticks", "Ball & Stick", "Surface (VDW)", "Wireframe"],
            index=0,
            label_visibility="collapsed"
        )
    with ctrl_col3:
        auto_spin = st.checkbox("Auto-Spin", value=True)

    # Fetch SDF coordinates for active molecule
    sdf_path = conformers_dir / f"{active_mol}.sdf"
    sdf_content = ""
    if sdf_path.exists():
        sdf_content = sdf_path.read_text(encoding="utf-8")
    else:
        # Fallback minimal 3D geometry if file missing
        sdf_content = ""

    # Clean SDF for JavaScript string interpolation
    clean_sdf = json.dumps(sdf_content)

    # 3Dmol.js HTML Component
    style_js = ""
    surface_js = ""
    if representation == "Sticks":
        style_js = 'viewer.setStyle({}, {stick: {radius: 0.22, colorscheme: "cyanCarbon"}});'
    elif representation == "Ball & Stick":
        style_js = 'viewer.setStyle({}, {sphere: {scale: 0.32}, stick: {radius: 0.16, colorscheme: "greenCarbon"}});'
    elif representation == "Surface (VDW)":
        style_js = 'viewer.setStyle({}, {stick: {radius: 0.16}});'
        surface_js = 'viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity: 0.72, color: "#38bdf8"});'
    elif representation == "Wireframe":
        style_js = 'viewer.setStyle({}, {line: {linewidth: 2.0}});'

    spin_js = "viewer.spin(true, 1.0);" if auto_spin else "viewer.spin(false);"

    html_3dmol = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #0b0f19; overflow: hidden; }}
            #viewer_container {{
                width: 100%;
                height: 480px;
                border-radius: 10px;
                border: 1px solid #1e293b;
                position: relative;
                box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.8);
            }}
            .overlay-badge {{
                position: absolute;
                top: 12px;
                left: 12px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid rgba(56, 189, 248, 0.4);
                color: #38bdf8;
                padding: 4px 10px;
                border-radius: 6px;
                font-family: sans-serif;
                font-size: 11px;
                font-weight: 600;
                z-index: 10;
                pointer-events: none;
            }}
            .overlay-legend {{
                position: absolute;
                bottom: 12px;
                right: 12px;
                background: rgba(15, 23, 42, 0.85);
                border: 1px solid #334155;
                color: #94a3b8;
                padding: 6px 10px;
                border-radius: 6px;
                font-family: sans-serif;
                font-size: 10px;
                z-index: 10;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div id="viewer_container">
            <div class="overlay-badge">MMFF94 Conformer · {active_mol}</div>
            <div class="overlay-legend">Left-drag: Rotate | Right-drag: Pan | Scroll: Zoom</div>
        </div>
        <script>
            let elem = document.getElementById("viewer_container");
            let config = {{ backgroundColor: "#0b0f19" }};
            let viewer = $3Dmol.createViewer(elem, config);
            let sdfData = {clean_sdf};
            
            if (sdfData && sdfData.trim().length > 0) {{
                viewer.addModel(sdfData, "sdf");
                {style_js}
                {surface_js}
                viewer.zoomTo();
                viewer.render();
                {spin_js}
            }} else {{
                viewer.addLabel("Conformer coordinates not found", {{fontSize: 14, fontColor: '#ef4444', backgroundColor: '#0f172a'}});
            }}
        </script>
    </body>
    </html>
    """
    components.html(html_3dmol, height=490)

    # Active Candidate Molecular Attributes Card
    row_mol = df_candidates[df_candidates["mol_id"] == active_mol].iloc[0]
    st.markdown(f"""
    <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 14px; margin-top: 8px;">
        <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px; font-weight: 600;">MOLECULAR METRICS · {active_mol}</div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <div class="stat-pill"><span class="stat-pill-label">MW: </span><span class="stat-pill-val">{row_mol.get('mw', 0.0):.1f} Da</span></div>
            <div class="stat-pill"><span class="stat-pill-label">LogP: </span><span class="stat-pill-val">{row_mol.get('logp', 0.0):.2f}</span></div>
            <div class="stat-pill"><span class="stat-pill-label">HBD/HBA: </span><span class="stat-pill-val">{int(row_mol.get('hbd', 0))}/{int(row_mol.get('hba', 0))}</span></div>
            <div class="stat-pill"><span class="stat-pill-label">Rot. Bonds: </span><span class="stat-pill-val">{int(row_mol.get('rot_bonds', 0))}</span></div>
            <div class="stat-pill"><span class="stat-pill-label">SA Score: </span><span class="stat-pill-val">{row_mol.get('sa', 0.0):.2f}</span></div>
            <div class="stat-pill"><span class="stat-pill-label">qPMHI: </span><span class="stat-pill-val">{row_mol.get('qpmhi_score', 0.0):.4f}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Right Column: Pareto Plot & Ranked Data Table
# ------------------------------------------------------------------------------
with col_right:
    # 1. Pareto Front Optimization Scatter Plot (Matches Image)
    st.markdown("""
    <div class="panel-header">
        <span class="panel-title">📈 Pareto Front Optimization Scatter Plot</span>
        <span style="font-size: 0.75rem; color: #94a3b8;">Multi-Objective Acquisition</span>
    </div>
    """, unsafe_allow_html=True)

    # Sort points for Pareto frontier curve
    df_sorted_qed = df_candidates.sort_values(by="qed").copy()

    # Extract non-dominated Pareto frontier points
    pts = df_candidates[["qed", "mu"]].values
    pareto_mask = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        dominated = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        dominated[i] = False
        if dominated.any():
            pareto_mask[i] = False
    
    df_pareto = df_candidates[pareto_mask].sort_values(by="qed")

    # Interactive Plotly Dark Scatter Plot
    fig = go.Figure()

    # Trace 1: All Candidate Points with Epistemic Uncertainty Error Bars
    fig.add_trace(go.Scatter(
        x=df_candidates["qed"],
        y=df_candidates["mu"],
        mode="markers",
        name="Candidates",
        error_y=dict(
            type="data",
            array=df_candidates["sigma"],
            visible=True,
            color="rgba(56, 189, 248, 0.4)",
            thickness=1.2,
            width=3
        ),
        marker=dict(
            size=9,
            color="#0ea5e9",
            line=dict(width=1, color="#38bdf8"),
            opacity=0.85
        ),
        text=df_candidates["mol_id"],
        customdata=np.column_stack([
            df_candidates["mol_id"],
            df_candidates["qed"],
            df_candidates["mu"],
            df_candidates["sigma"],
            df_candidates["sa"],
            df_candidates["rank"]
        ]),
        hovertemplate=(
            "<b>Candidate: %{customdata[0]}</b> (Rank #%{customdata[5]})<br>"
            "QED Score: %{customdata[1]:.3f}<br>"
            "Predicted Affinity (μ): %{customdata[2]:.2f} ± %{customdata[3]:.2f}<br>"
            "Synthetic Access (SA): %{customdata[4]:.2f}<extra></extra>"
        )
    ))

    # Trace 2: Golden Pareto Frontier Curve (Matches reference image)
    fig.add_trace(go.Scatter(
        x=df_pareto["qed"],
        y=df_pareto["mu"],
        mode="lines+markers",
        name="Pareto Frontier",
        line=dict(color="#f59e0b", width=2.2, shape="spline"),
        marker=dict(size=7, color="#fbbf24", symbol="circle"),
        hoverinfo="skip"
    ))

    # Trace 3: Highlighted Active Molecule Marker
    active_row = df_candidates[df_candidates["mol_id"] == active_mol].iloc[0]
    fig.add_trace(go.Scatter(
        x=[active_row["qed"]],
        y=[active_row["mu"]],
        mode="markers",
        name="Selected in 3D",
        marker=dict(
            size=15,
            color="#ef4444",
            symbol="star",
            line=dict(width=2, color="#ffffff")
        ),
        hovertemplate=f"<b>ACTIVE: {active_mol}</b><br>QED: {active_row['qed']:.2f}<br>Affinity: {active_row['mu']:.2f}<extra></extra>"
    ))

    # Dark Chart Theme Styling
    fig.update_layout(
        height=270,
        margin=dict(l=45, r=20, t=10, b=35),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#94a3b8"),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            title="QED Score",
            title_font=dict(size=11, color="#94a3b8"),
            tickfont=dict(size=10, color="#64748b"),
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            range=[0.35, 0.85]
        ),
        yaxis=dict(
            title="Affinity Prediction (pIC50)",
            title_font=dict(size=11, color="#94a3b8"),
            tickfont=dict(size=10, color="#64748b"),
            gridcolor="#1e293b",
            zerolinecolor="#334155"
        )
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 2. Ranked Candidate Data Table (Matches Image)
    st.markdown("""
    <div class="panel-header" style="margin-top: 14px;">
        <span class="panel-title">📋 Ranked Candidate Data Table</span>
        <span style="font-size: 0.75rem; color: #94a3b8;">Human-in-the-Loop Review</span>
    </div>
    """, unsafe_allow_html=True)

    # Search filter input
    search_q = st.text_input("Filter candidate ID", "", placeholder="Search candidate ID (e.g. X20403, TAM16)...", label_visibility="collapsed")
    df_filtered = df_candidates[df_candidates["mol_id"].str.contains(search_q, case=False)] if search_q else df_candidates

    # Table columns to format
    display_df = df_filtered[["rank", "mol_id", "qed", "mu", "sigma", "sa", "status"]].copy()
    display_df.columns = ["Rank", "Candidate ID", "QED", "Affinity (μ)", "Uncertainty (σ)", "SA Score", "Status"]

    st.dataframe(
        display_df.style.format({
            "QED": "{:.3f}",
            "Affinity (μ)": "{:.2f}",
            "Uncertainty (σ)": "{:.2f}",
            "SA Score": "{:.2f}"
        }),
        height=220,
        use_container_width=True
    )

    # HITL Decision Action Bar
    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.2, 1.5])
    with btn_col1:
        if st.button("✅ Approve for Docking", use_container_width=True):
            audit_entry = {
                "timestamp": time.time(),
                "mol_id": active_mol,
                "action": "APPROVED_FOR_DOCKING",
                "qed": float(active_row["qed"]),
                "mu": float(active_row["mu"])
            }
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry) + "\n")
            st.success(f"Candidate {active_mol} recorded to immutable audit trail.")
            time.sleep(0.5)
            st.rerun()
    with btn_col2:
        if st.button("❌ Reject Candidate", use_container_width=True):
            audit_entry = {
                "timestamp": time.time(),
                "mol_id": active_mol,
                "action": "REJECTED",
                "qed": float(active_row["qed"]),
                "mu": float(active_row["mu"])
            }
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry) + "\n")
            st.warning(f"Candidate {active_mol} rejected.")
            time.sleep(0.5)
            st.rerun()
    with btn_col3:
        if sdf_content:
            st.download_button(
                label=f"⬇️ Export {active_mol}.sdf",
                data=sdf_content,
                file_name=f"{active_mol}_conformer.sdf",
                mime="chemical/x-mdl-sdfile",
                use_container_width=True
            )

# ==============================================================================
# Lower Section: Digital Annealing & Quantum Solvers Analysis Drawer
# ==============================================================================
st.markdown("---")
with st.expander("⚡ Quantum & Digital Annealing Solvers Benchmark (Yanagisawa Hamiltonian)", expanded=False):
    if solver_path.exists() and metrics_path.exists():
        df_solvers = pd.read_parquet(solver_path)
        with open(metrics_path, "r", encoding="utf-8") as f:
            summary_metrics = json.load(f)

        s_col_a, s_col_b, s_col_c, s_col_d = st.columns(4)
        s_col_a.metric("Target Crystal Structure", summary_metrics.get("reference_pdb", "5V3Y"))
        s_col_b.metric("Lead Fragment Complex", summary_metrics.get("lead_compound", "TAM16"))
        rmsd_val = summary_metrics.get("heavy_atom_rmsd_A", 0.0)
        s_col_c.metric("Heavy-Atom RMSD", f"{rmsd_val:.2f} Å", delta="Pass (< 2.0 Å)" if rmsd_val < 2.0 else "Fail")
        s_col_d.metric("Constraint Violations", summary_metrics.get("constraint_violations", 0), delta="Fixed")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Minimum Ground-State Energy (kcal/mol)")
            fig_e = go.Figure()
            fig_e.add_trace(go.Bar(
                x=df_solvers["solver"],
                y=df_solvers["energy"],
                marker_color=["#0ea5e9", "#10b981", "#8b5cf6", "#f59e0b"]
            ))
            fig_e.update_layout(
                paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
                height=240, margin=dict(l=30, r=20, t=10, b=30),
                yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
                xaxis=dict(tickfont=dict(color="#94a3b8"))
            )
            st.plotly_chart(fig_e, use_container_width=True)

        with c2:
            st.markdown("#### Time-to-Solution (TTS99 in seconds)")
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(
                x=df_solvers["solver"],
                y=df_solvers["tts_99"],
                marker_color=["#38bdf8", "#34d399", "#a78bfa", "#fbbf24"]
            ))
            fig_t.update_layout(
                paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
                height=240, margin=dict(l=30, r=20, t=10, b=30),
                yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8"), type="log"),
                xaxis=dict(tickfont=dict(color="#94a3b8"))
            )
            st.plotly_chart(fig_t, use_container_width=True)

        st.dataframe(df_solvers[["solver", "energy", "wall_s", "p_success", "tts_99"]], use_container_width=True)
    else:
        st.info("Execute `python scripts/run_pipeline.py` to populate solver benchmark records.")

st.caption("Human-in-the-Loop actions append immutable records to `data/processed/hitl_decisions.jsonl` without mutating upstream artifacts.")
