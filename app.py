"""WRAPA-ViT results explorer (Streamlit).

Run:  streamlit run app.py
All numbers come from the project-wrapa notebook runs reported in the paper.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="WRAPA-ViT: Geometry-Aware Wafer Defect Classification", page_icon="🔬", layout="wide")

# ----------------------------------------------------------------- data
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
DIST = pd.DataFrame({
    "Class": ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc", "Scratch", "Random", "Donut", "Near-full"],
    "Total": [147431, 9680, 5189, 4294, 3593, 1193, 866, 555, 149],
    "Train": [103202, 6776, 3632, 3006, 2515, 835, 606, 389, 104],
    "Test": [22115, 1452, 779, 644, 539, 179, 130, 83, 22],
    "Loss weight": [0.130, 1.985, 3.704, 4.475, 5.349, 16.110, 22.198, 34.580, 129.343],
})
# Values transcribed from the corrected Phase I report (Tables 4.3–4.5).
ABL = pd.DataFrame({
    "Variant": ["Cartesian (XY) only", "Radial only", "Angular only", "Radial + Angular", "WRAPA (learnable XY + R + A)"],
    "Accuracy": [.9797, .9799, .9798, .9802, .9866],
    "Macro-F1": [.5885, .5753, .5659, .5698, .6932],
    "Weighted-F1": [.9784, .9783, .9780, .9785, .9834]
})
WM_SUMMARY = {"Accuracy": .9866, "Macro-F1": .6932, "Weighted-F1": .9834}
CHECKPOINTS = pd.DataFrame({"Seed": [42, 123, 2024], "Best epoch": [9, 9, 7],
                            "Best validation Macro-F1": [.6839, .7171, .6787],
                            "Parameters": [1224121] * 3})
WM_GATE_SEED = pd.DataFrame({"Seed": [42, 123, 2024],
                             "XY": [1.0139, .9790, .9992],
                             "Radial": [1.0209, 1.0123, 1.0531],
                             "Angular": [.9652, 1.0087, .9477]})
GATE_WM = np.array([[.9746,1.0091,1.0163],[.9680,1.0406,.9914],[.9908,1.0403,.9689],
                    [1.0191,1.0414,.9395],[.9438,1.0635,.9927],[1.0879,.9777,.9345]])
GATE_MIX = np.array([[1.0019,.9878,1.0103],[1.0173,.9995,.9832],[1.0075,.9801,1.0123],
                     [.9957,.9615,1.0428],[.9834,.9062,1.1103],[1.0118,.8864,1.1018]])
MIX_PC = pd.DataFrame({"Label": ["Center","Donut","Edge-Loc","Edge-Ring","Loc","Near-Full","Scratch","Random"],
                       "Precision": [.9959,.9994,.9953,.9846,.9954,1.0,.9514,.9773],
                       "Recall": [.9879,.9989,.9758,.9983,.9693,1.0,.9582,1.0]})
MIX_PC["F1"] = 2 * MIX_PC.Precision * MIX_PC.Recall / (MIX_PC.Precision + MIX_PC.Recall)
MIX_TH = pd.DataFrame({"Threshold": [.30,.35,.40,.45,.50,.55,.60,.65,.70],
                       "Macro-F1": [.9786,.9820,.9851,.9856,.9867,.9867,.9864,.9863,.9860],
                       "Micro-F1": [.9801,.9808,.9815,.9817,.9819,.9820,.9816,.9814,.9809],
                       "Exact match": [.9164,.9199,.9228,.9239,.9257,.9262,.9243,.9237,.9214]})
MIX_OVERALL = {"Exact-match accuracy": "92.57%", "Macro-F1": "98.67%", "Micro-F1": "98.19%",
               "Macro precision / recall": "98.74% / 98.61%", "Hamming loss": "0.0103",
               "Macro ROC-AUC / PR-AUC": "99.76% / 99.72%"}
# Standalone report-based dashboard: no external CSV files required.
MIXTR = pd.DataFrame({
    "epoch": [1,2,3,5,8,10,12,15],
    "train_loss": [.4964,.2977,.2180,.1242,.0755,.0562,.0424,.0340],
    "val_loss": [.3669,.2177,.1588,.0876,.0617,.0576,.0449,.0405],
    "train_macro_f1": [.6284,.7932,.8579,.9300,.9576,.9688,.9772,.9857],
    "val_macro_f1": [.7156,.8482,.8802,.9518,.9677,.9660,.9763,.9790],
    "val_micro_f1": [.7273,.8602,.9112,.9580,.9674,.9720,.9789,.9810]
})
FIXED_COLORS = {"A4_ARoPE": "#222222", "A5_ARoPE_Radial": "#1f77b4", "A6_ARoPE_Angular": "#d95f02",
                "A7_ARoPE_Radial_Angular": "#2ca02c"}
NAMES = {"A4_ARoPE": "A4 XY", "A5_ARoPE_Radial": "A5 XY+Radial", "A6_ARoPE_Angular": "A6 XY+Angular",
         "A7_ARoPE_Radial_Angular": "A7 XY+Rad.+Ang."}
pct = lambda v: f"{v*100:.2f}%"

# ----------------------------------------------------------------- pages
st.markdown("""<style>
.stApp {background: #f5f8fc; color: #14233b;}
[data-testid="stSidebar"] {background: #10233f;}
[data-testid="stSidebar"] * {color: #eef7ff !important;}
h1,h2,h3 {color: #102b4b !important;}
[data-testid="stMetric"] {background: white; padding: 17px; border-radius: 13px;
 border: 1px solid #dbe7f0; box-shadow: 0 2px 9px #102b4b10;}
</style>""", unsafe_allow_html=True)

def page_overview():
    st.title("WRAPA-ViT: Geometry-Aware Wafer Defect Classification")
    st.caption("Weighted Rotary Positional Attention with learnable Cartesian, radial and angular geometry | Phase I | Kavitha R V")
    st.markdown("**Research focus:** Learnable wafer-specific geometry in rotary attention for single-label WM-811K and multi-label MixedWM38 recognition.")
    c = st.columns(4)
    c[0].metric("WM-811K accuracy", "98.66%")
    c[1].metric("WM-811K Macro-F1", "69.32%")
    c[2].metric("WM-811K Weighted-F1", "98.34%")
    c[3].metric("MixedWM38 Macro-F1", "98.67%")
    st.subheader("What makes WRAPA different?")
    st.markdown("""WRAPA represents wafer patch positions through **Cartesian (XY)**,
    **radial distance** and **angular direction**, then integrates these signals into rotary
    query/key attention. **Block-wise learnable geometry weights** allow the six Transformer
    blocks to redistribute attention among the three spatial views. The weights sum to three.
    """)
    st.subheader("Evaluation at a glance")
    a,b = st.columns(2)
    with a:
        st.markdown("**WM-811K · single-label, nine classes**")
        st.dataframe(pd.DataFrame([{"Metric": k, "Reported value": pct(v)} for k,v in WM_SUMMARY.items()]), hide_index=True, width="stretch")
        st.caption("Report Table 4.3: mean over seeds 42, 123 and 2024.")
    with b:
        st.markdown("**MixedWM38 · multi-label, eight base defects**")
        st.dataframe(pd.DataFrame([{"Metric": k, "Reported value": v} for k,v in MIX_OVERALL.items()]), hide_index=True, width="stretch")
        st.caption("Report Table 4.6: test set n=5,703; decision threshold 0.5.")
    st.info("Values are transcribed from the corrected Phase I report. WM-811K mean test metrics and best validation checkpoint scores are distinct measures; do not compare them directly.")


def page_dataset():
    st.header("WM-811K · Dataset and task")
    st.metric("Total wafer maps", "811,457")
    st.metric("Approximately labelled maps", "173,000")
    st.markdown("**Task:** nine-class single-label classification: eight defect patterns plus `none`. **Input:** 64 × 64 grayscale images. **Output:** nine-way softmax. **Evaluation:** held-out split with three random seeds (42, 123, 2024).")
    st.markdown("**Defect labels:** " + ", ".join(CLASSES))
    st.info("The labelled subset is strongly imbalanced. Macro-F1 is therefore reported alongside overall accuracy and Weighted-F1.")
    st.caption("Source: corrected Phase I report, Tables 4.1–4.3. Legacy 224 × 224 class counts have been removed because they do not describe the corrected 64 × 64 protocol.")


def page_ablation():
    st.header("WM-811K · Controlled geometry ablation")
    st.caption("Corrected Phase I report, Table 4.5: mean over three seeds. Percentages are displayed as scores.")
    metrics = st.multiselect("Compare metrics", ["Accuracy", "Macro-F1", "Weighted-F1"], default=["Macro-F1", "Accuracy"])
    if metrics:
        d = ABL.melt("Variant", metrics, var_name="Metric", value_name="Score")
        fig = px.bar(d, x="Variant", y="Score", color="Metric", barmode="group", text_auto=".3f",
                     color_discrete_sequence=["#087f8c", "#193b69", "#edaa4b"])
        fig.update_yaxes(range=[0,1], tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    st.dataframe(ABL.style.format({m:"{:.2%}" for m in ["Accuracy","Macro-F1","Weighted-F1"]}), hide_index=True, width="stretch")
    st.success("In the reported controlled ablation, the full learnable WRAPA variant achieves 69.32% Macro-F1 versus 58.85% for Cartesian-only attention (+10.47 percentage points).")


def page_seeds():
    st.header("WM-811K · Three-seed checkpoint analysis")
    st.caption("Corrected Phase I report, Table 4.4: best *validation* Macro-F1 at the selected checkpoint.")
    st.metric("Best validation Macro-F1 · mean ± SD", "69.32 ± 2.09%")
    fig = px.bar(CHECKPOINTS, x="Seed", y="Best validation Macro-F1", text_auto=".2%",
                 color_discrete_sequence=["#087f8c"])
    fig.update_yaxes(range=[0, .85], tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(CHECKPOINTS.style.format({"Best validation Macro-F1":"{:.2%}"}), hide_index=True, width="stretch")
    st.info("The checkpoint scores above are validation results. The 98.66% WM-811K accuracy on the Overview page is the report's mean accuracy and is not a validation Macro-F1 score.")


def page_gates():
    st.header("Learned geometry weights across six Transformer blocks")
    st.caption("w = 3 × softmax(logits); the three geometry weights sum to three per block.")
    a,b=st.columns(2)
    for col, M, title in [(a, GATE_WM, "WM-811K · 3-seed block mean"), (b, GATE_MIX, "MixedWM38")]:
        fig=px.imshow(M.T, x=[f"Block {i}" for i in range(1,7)], y=["XY","Radial","Angular"],
                      text_auto=".2f", color_continuous_scale="RdBu_r", zmin=.85,zmax=1.15,aspect="auto")
        fig.update_layout(title=title, height=340)
        col.plotly_chart(fig,width="stretch")
    st.markdown("**WM-811K block-averaged geometry weights by seed**")
    st.dataframe(WM_GATE_SEED,hide_index=True,width="stretch")
    st.caption("Report Tables 4.11, 4.13 and 4.14. MixedWM38 exhibits decreasing radial and increasing angular weight toward deeper blocks.")


def page_mixed():
    st.header("MixedWM38 multi-label results")
    cols = st.columns(len(MIX_OVERALL))
    for c, (k, v) in zip(cols, MIX_OVERALL.items()):
        c.metric(k, v)
    tab1, tab2, tab3 = st.tabs(["Training", "Per label", "Threshold sensitivity"])
    with tab1:
        a, b = st.columns(2)
        a.plotly_chart(px.line(MIXTR, x="epoch", y=["train_loss", "val_loss"], markers=True,
                               title="BCE loss"), width='stretch')
        b.plotly_chart(px.line(MIXTR, x="epoch", y=["train_macro_f1", "val_macro_f1"], markers=True,
                               title="Macro-F1"), width='stretch')
    with tab2:
        long = MIX_PC.melt("Label", ["Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Label", y="Score", color="Metric", barmode="group", range_y=[0.93, 1.005])
        st.plotly_chart(fig, width='stretch')
    with tab3:
        fig = px.line(MIX_TH, x="Threshold", y=["Macro-F1", "Micro-F1", "Exact match"], markers=True)
        st.plotly_chart(fig, width='stretch')
        st.caption("Macro-F1 stays between 97.86% and 98.67% for thresholds 0.30–0.70.")
    st.caption("Source: corrected Phase I report, Tables 4.6–4.10. No baseline was trained on MixedWM38, so no superiority claim is made.")


def page_geometry():
    st.header("Interactive: how the rotary phases look on a patch grid")
    st.caption("Illustration of the phase definitions in the paper with hand-set coefficients — not learned weights.")
    c = st.columns(3)
    G = c[0].slider("Patch grid size G", 4, 14, 14)
    comps = c[1].multiselect("Phase terms", ["XY", "Radial", "Angular"], default=["XY", "Radial", "Angular"])
    w = {k: c[2].slider(f"weight {k}", 0.0, 3.0, 1.0, 0.05) for k in ["XY", "Radial", "Angular"]}
    a1, a2, a3 = st.columns(3)
    ax_, ay_ = a1.slider("α_x", -3.0, 3.0, 1.0, 0.1), a1.slider("α_y", -3.0, 3.0, 1.0, 0.1)
    ar_ = a2.slider("α_r", -6.0, 6.0, 3.0, 0.1)
    as_, ac_ = a3.slider("α_s", -3.0, 3.0, 1.0, 0.1), a3.slider("α_c", -3.0, 3.0, 1.0, 0.1)
    idx = np.arange(G)
    col, row = np.meshgrid(idx, idx)
    x, y = col / (G - 1), row / (G - 1)
    xt, yt = (col - (G - 1) / 2) / ((G - 1) / 2), (row - (G - 1) / 2) / ((G - 1) / 2)
    r = np.sqrt(xt**2 + yt**2) / np.sqrt(2)
    th = np.arctan2(yt, xt)
    phi_xy = ax_ * x + ay_ * y
    phi_r = ar_ * r
    phi_a = as_ * np.sin(th) + ac_ * np.cos(th)
    phi = np.zeros_like(x)
    if "XY" in comps: phi = phi + w["XY"] * phi_xy
    if "Radial" in comps: phi = phi + w["Radial"] * phi_r
    if "Angular" in comps: phi = phi + w["Angular"] * phi_a
    q = st.slider("Query patch (row, col index)", 0, G * G - 1, (G // 2) * G + G // 2)
    qr, qc = divmod(q, G)
    bias = np.cos(phi - phi[qr, qc])
    m1, m2 = st.columns(2)
    f1 = px.imshow(phi, color_continuous_scale="Viridis", title="Combined phase φₙ", aspect="equal")
    m1.plotly_chart(f1, width='stretch')
    f2 = px.imshow(bias, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="equal",
                   title="cos(φₙ − φ_query): relative-phase similarity")
    f2.add_scatter(x=[qc], y=[qr], mode="markers", marker=dict(color="black", size=11, symbol="x"), name="query")
    m2.plotly_chart(f2, width='stretch')
    st.caption("Because both q and k are rotated by the same scalar phase, the attention score depends on the phase "
               "difference φₙ − φₘ. The right panel shows that difference as a cosine from the chosen query patch.")



def page_wafer_gallery():
    st.header("Real Wafer-Map Image Gallery")
    st.caption("Explore authentic wafer-map samples from WM-811K or MixedWM38. Images are displayed as supplied; no defect labels or model predictions are invented.")
    st.info("To make this gallery permanent, add your original wafer-map images to a folder named `wafer_images` beside `app.py` in GitHub. Name each image after its verified defect class, such as `Center_01.png` or `Scratch_02.png`.")

    folder = HERE / "wafer_images"
    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
    saved = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_ext) if folder.exists() else []
    uploaded = st.file_uploader("Preview wafer-map images from your computer", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
                                help="Uploaded files are previewed during this session; they are not automatically committed to GitHub.")
    images = [(f.name, f) for f in saved] + [(f.name, f) for f in (uploaded or [])]
    if not images:
        st.warning("No wafer images are available yet. Upload original WM-811K/MixedWM38 samples above, or commit images to the `wafer_images/` folder.")
        st.markdown("**Suggested verified categories:** Center · Donut · Edge-Loc · Edge-Ring · Loc · Near-full · Random · Scratch · none (WM-811K).")
        return

    st.success(f"{len(images)} image(s) available")
    names = sorted(set(name.split("_")[0] for name, _ in images))
    selection = st.selectbox("Filter by filename prefix", ["All images"] + names)
    chosen = [(name, obj) for name, obj in images if selection == "All images" or name.split("_")[0] == selection]
    columns = st.columns(4)
    for i, (name, obj) in enumerate(chosen):
        with columns[i % 4]:
            st.image(obj, caption=name, use_container_width=True)
    st.caption("Source labels are inferred only from filenames for filtering. Verify each class against the original dataset metadata. This gallery is not a live factory feed or an inference demo.")


PAGES = {
    "Research overview": page_overview,
    "WM-811K dataset": page_dataset,
    "Wafer image gallery": page_wafer_gallery,
    "Geometry ablation": page_ablation,
    "Three-seed evaluation": page_seeds,
    "Learned geometry weights": page_gates,
    "MixedWM38 results": page_mixed,
    "Interactive phase explorer": page_geometry,
}
st.sidebar.title("WRAPA-ViT")
st.sidebar.caption("Geometry-Aware Wafer Defect Classification")
choice = st.sidebar.radio("Explore research", list(PAGES))
st.sidebar.divider()
st.sidebar.caption("Source: corrected October 2026 Phase I report. Reported figures are not independently re-computed.")
PAGES[choice]()
