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
st.set_page_config(page_title="WRAPA-ViT Results Explorer", page_icon="🔬", layout="wide")

# ----------------------------------------------------------------- data
CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
DIST = pd.DataFrame({
    "Class": ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc", "Scratch", "Random", "Donut", "Near-full"],
    "Total": [147431, 9680, 5189, 4294, 3593, 1193, 866, 555, 149],
    "Train": [103202, 6776, 3632, 3006, 2515, 835, 606, 389, 104],
    "Test": [22115, 1452, 779, 644, 539, 179, 130, 83, 22],
    "Loss weight": [0.130, 1.985, 3.704, 4.475, 5.349, 16.110, 22.198, 34.580, 129.343],
})
ABL = pd.DataFrame({
    "Variant": ["A4 XY", "A5 XY+Radial", "A6 XY+Angular", "A7 XY+Rad.+Ang."],
    "Accuracy": [.9387, .9476, .9308, .9471], "Macro-F1": [.7101, .7602, .7189, .7564],
    "Weighted-F1": [.9442, .9518, .9394, .9521], "Balanced acc.": [.7484, .7918, .7697, .7969]})
SEED3 = pd.DataFrame({
    "Variant": ["A4 XY", "A5 +Radial", "A7 +Rad.+Ang.", "A8 adaptive"],
    "Accuracy": [.901348, .886598, .894281, .8966], "Accuracy SD": [.028955, .0265, .0252, .0061],
    "Macro-F1": [.695073, .721848, .732417, .6924], "Macro-F1 SD": [.010138, .021554, .016541, .0287],
    "Weighted-F1": [.91967, .911825, .917622, .9134], "Weighted-F1 SD": [.0184, .0186, .0176, .0052],
    "Balanced acc.": [.784336, .84717, .845634, .7860], "Balanced acc. SD": [.03676, .016625, .027172, .0241]})
A8_SEEDS = pd.DataFrame({"Seed": [42, 123, 2024], "Accuracy": [.8962, .9028, .8907],
                         "Macro-F1": [.6809, .7251, .6712], "Weighted-F1": [.9132, .9186, .9083],
                         "Balanced acc.": [.7989, .8010, .7582]})
ROT = {  # metric order: accuracy, macro-F1, balanced accuracy ; angles 0/90/180/270
    "A4 XY": ([.9387, .7068, .7106, .7656], [.7101, .4843, .5216, .5102], [.7484, .5933, .6154, .5628]),
    "A5 XY+Radial": ([.9476, .6936, .6690, .7936], [.7602, .5650, .5570, .6018], [.7918, .6871, .6792, .6688]),
    "A6 XY+Angular": ([.9308, .6425, .6394, .7382], [.7189, .5155, .4938, .5458], [.7697, .6324, .6425, .6203]),
    "A7 XY+Rad.+Ang.": ([.9471, .7279, .6792, .8040], [.7564, .5988, .5698, .5980], [.7969, .7221, .6902, .6760])}
CM = np.array([[20234,141,1,911,104,179,0,25,520],[37,588,1,4,0,7,0,3,4],[0,0,73,1,0,4,0,1,4],
               [58,4,0,642,21,33,0,5,16],[2,0,0,41,1404,2,0,1,2],[77,15,12,61,0,292,0,8,74],
               [0,0,0,0,0,0,22,0,0],[0,0,3,6,2,3,1,112,3],[78,4,1,17,0,24,0,0,55]])
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
HIST = pd.read_csv(HERE / "wm811k_ablation_history.csv")
A8PC = pd.read_csv(HERE / "a8_per_class_3seed.csv")
MIXTR = pd.read_csv(HERE / "mixedwm38_training.csv")
FIXED_COLORS = {"A4_ARoPE": "#222222", "A5_ARoPE_Radial": "#1f77b4", "A6_ARoPE_Angular": "#d95f02",
                "A7_ARoPE_Radial_Angular": "#2ca02c"}
NAMES = {"A4_ARoPE": "A4 XY", "A5_ARoPE_Radial": "A5 XY+Radial", "A6_ARoPE_Angular": "A6 XY+Angular",
         "A7_ARoPE_Radial_Angular": "A7 XY+Rad.+Ang."}
pct = lambda v: f"{v*100:.2f}%"

# ----------------------------------------------------------------- pages
def page_overview():
    st.title("WRAPA-ViT results explorer")
    st.caption("Wafer-specific Radial–Angular Positional Attention for wafer-map defect recognition "
               "· Kavitha R V, S. Prasanna Devi, L. Jabasheela")
    c = st.columns(4)
    c[0].metric("WM-811K macro-F1 (A7, 3 seeds)", "97.2 ± 1.7 %", "+3.7 pts vs A4")
    c[1].metric("WM-811K balanced acc. (A7)", "84.6 ± 2.7 %", "+6.1 pts vs A4")
    c[2].metric("Adaptive gate (A8) macro-F1", "69.2 ± 2.9 %", "no gain vs A4", delta_color="off")
    c[3].metric("MixedWM38 macro-F1", "98.67 %")
    st.markdown("""
**What this app shows.** Every chart is built from the notebook results reported in the paper.
Use the sidebar to move between the experiments. The last page is an interactive illustration of how the
Cartesian, radial and angular rotary phases look on a patch grid.

**Honest reading of the results**
- Radial and radial–angular phases improve macro-F1 and balanced accuracy over the Cartesian baseline; margins over three seeds are modest and overlap.
- The learned geometry gate (A8) did **not** beat the fixed radial–angular phase on WM-811K, and its weights stay near uniform.
- *Scratch* is the weakest class on both datasets. No baseline was trained on MixedWM38.
- Single-run ablation and three-seed tables come from separate runs; do not compare them numerically.
""")


def page_dataset():
    st.header("WM-811K class distribution")
    log = st.toggle("Log scale", value=True)
    fig = px.bar(DIST, x="Class", y="Total", text="Total", color_discrete_sequence=["#4c72b0"])
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_yaxes(type="log" if log else "linear", title="Wafer maps")
    st.plotly_chart(fig, width='stretch')
    c1, c2 = st.columns(2)
    c1.dataframe(DIST, hide_index=True, width='stretch')
    fig2 = px.bar(DIST, x="Class", y="Loss weight", text="Loss weight", color_discrete_sequence=["#dd8452"])
    fig2.update_yaxes(type="log", title="Inverse-frequency weight (log)")
    c2.plotly_chart(fig2, width='stretch')
    st.info("'none' is about 85% of the test set, so accuracy mostly reflects the majority class; "
            "macro-F1 and balanced accuracy are the primary metrics.")


def page_ablation():
    st.header("Geometry ablation on WM-811K")
    tab1, tab2 = st.tabs(["Single run", "Training curves"])
    with tab1:
        metrics = st.multiselect("Metrics", ["Accuracy", "Macro-F1", "Weighted-F1", "Balanced acc."],
                                 default=["Accuracy", "Macro-F1", "Balanced acc."])
        long = ABL.melt("Variant", metrics, var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Variant", y="Score", color="Metric", barmode="group", text_auto=".3f",
                     range_y=[0.6, 1.0])
        st.plotly_chart(fig, width='stretch')
        st.dataframe(ABL.style.format({c: "{:.2%}" for c in ABL.columns[1:]}), hide_index=True,
                     width='stretch')
        st.caption("Single run, 224×224 input. A5 and A7 are within the seed-to-seed spread of each other.")
    with tab2:
        metric = st.radio("Curve", ["val_f1", "train_loss", "val_loss", "val_acc"], horizontal=True)
        d = HIST.copy(); d["Variant"] = d.exp.map(NAMES)
        d = d[d.Variant.notna()]
        fig = px.line(d, x="epoch", y=metric, color="Variant", markers=True,
                      color_discrete_map={NAMES[k]: v for k, v in FIXED_COLORS.items()})
        st.plotly_chart(fig, width='stretch')


def page_seeds():
    st.header("Three-seed validation")
    metric = st.selectbox("Metric", ["Macro-F1", "Balanced acc.", "Accuracy", "Weighted-F1"])
    fig = go.Figure(go.Bar(x=SEED3.Variant, y=SEED3[metric], error_y=dict(type="data", array=SEED3[metric + " SD"]),
                           marker_color=["#222", "#1f77b4", "#2ca02c", "#d95f02"],
                           text=[f"{v*100:.1f}" for v in SEED3[metric]], textposition="outside"))
    fig.update_yaxes(range=[0.5, 1.0], title=f"{metric} (mean ± SD, 3 seeds)")
    st.plotly_chart(fig, width='stretch')
    show = SEED3.copy()
    for m in ["Accuracy", "Macro-F1", "Weighted-F1", "Balanced acc."]:
        show[m] = [f"{a*100:.2f} ± {b*100:.2f}" for a, b in zip(SEED3[m], SEED3[m + " SD"])]
    st.dataframe(show[["Variant", "Accuracy", "Macro-F1", "Weighted-F1", "Balanced acc."]], hide_index=True,
                 width='stretch')
    st.subheader("Adaptive gate (A8): per-seed results")
    st.dataframe(A8_SEEDS.style.format({c: "{:.2%}" for c in A8_SEEDS.columns[1:]}), hide_index=True,
                 width='stretch')
    st.warning("A8 uses a separate implementation (integer grid coordinates, coefficients initialised to 1) "
               "from A4–A7, so it is not a strictly paired comparison.")


def page_rotation():
    st.header("Robustness to test-time rotation")
    metric = st.radio("Metric", ["Accuracy", "Macro-F1", "Balanced accuracy"], horizontal=True)
    k = ["Accuracy", "Macro-F1", "Balanced accuracy"].index(metric)
    rows = [(v, a, ROT[v][k][i]) for v in ROT for i, a in enumerate([0, 90, 180, 270])]
    d = pd.DataFrame(rows, columns=["Variant", "Rotation", metric])
    fig = px.line(d, x="Rotation", y=metric, color="Variant", markers=True)
    fig.update_xaxes(tickvals=[0, 90, 180, 270], title="Test-time rotation (°)")
    st.plotly_chart(fig, width='stretch')
    drop = pd.DataFrame({v: [ROT[v][j][0] - np.mean(ROT[v][j][1:]) for j in range(3)] for v in ROT},
                        index=["Accuracy", "Macro-F1", "Balanced accuracy"]).T * 100
    drop.columns = [c + " drop (pts)" for c in drop.columns]
    st.subheader("Mean drop over 90°/180°/270°")
    st.dataframe(drop.round(1), width='stretch')
    st.caption("Models were trained on upright wafers; one checkpoint per variant. "
               "No model is rotation-invariant.")


def page_perclass():
    st.header("Per-class behaviour of the adaptive model (WM-811K)")
    tab1, tab2 = st.tabs(["Precision / recall / F1", "Confusion matrix"])
    with tab1:
        long = A8PC.melt("cls", ["precision", "recall", "f1"], var_name="Metric", value_name="Score")
        fig = px.bar(long, x="cls", y="Score", color="Metric", barmode="group", range_y=[0, 1.05])
        fig.update_xaxes(title="")
        st.plotly_chart(fig, width='stretch')
        st.caption("Mean over 3 seeds. Scratch F1 ≈ 0.13: low precision and recall.")
    with tab2:
        norm = st.toggle("Row-normalise", value=True)
        M = CM / CM.sum(1, keepdims=True) if norm else CM
        fig = px.imshow(M, x=CLASSES, y=CLASSES, text_auto=".2f" if norm else True,
                        color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color=""))
        fig.update_layout(height=620)
        st.plotly_chart(fig, width='stretch')
        st.caption("Seed 123. 44% of Scratch wafers are predicted as 'none'; 4.1% of 'none' wafers are labelled Edge-Loc.")


def page_gates():
    st.header("Learned geometry gate weights")
    st.caption("w = 3·softmax(ℓ); 1.0 is the uniform initialisation.")
    c1, c2 = st.columns(2)
    for col, M, t in [(c1, GATE_WM, "WM-811K (3-seed mean)"), (c2, GATE_MIX, "MixedWM38")]:
        fig = px.imshow(M.T, x=[f"B{i}" for i in range(1, 7)], y=["XY", "Radial", "Angular"],
                        color_continuous_scale="RdBu_r", zmin=0.85, zmax=1.15, text_auto=".2f", aspect="auto")
        fig.update_layout(title=t, height=330, coloraxis_showscale=False)
        col.plotly_chart(fig, width='stretch')
    st.info("All seed-averaged weights stay within 0.89–1.11, i.e. close to uniform, consistent with the "
            "finding that adaptive gating adds little on WM-811K.")


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
    st.caption("No baseline was trained on MixedWM38, so no superiority claim is made.")


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


PAGES = {"Overview": page_overview, "WM-811K dataset": page_dataset, "Geometry ablation": page_ablation,
         "Three-seed + adaptive gate": page_seeds, "Rotation robustness": page_rotation,
         "Per-class & confusion": page_perclass, "Learned gate weights": page_gates,
         "MixedWM38": page_mixed, "Phase explorer (interactive)": page_geometry}
st.sidebar.title("WRAPA-ViT")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("Data: project-wrapa notebook, as reported in the paper.")
PAGES[choice]()
