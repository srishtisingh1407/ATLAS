"""
GalaxEye Change Detection — Results Dashboard
Run with:  streamlit run code/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="GalaxEye Change Detection",
    page_icon="🛰️",
    layout="wide",
)

RUNS_ROOT  = Path(__file__).parent.parent / "runs"
CKPT_PATH  = RUNS_ROOT / "baseline_unet" / "checkpoints" / "best.pth"
HIST_PATH  = RUNS_ROOT / "baseline_unet" / "training_history.json"
SWEEP_PATH = RUNS_ROOT / "baseline_unet" / "threshold_sweep.json"
VAL_PATH   = RUNS_ROOT / "baseline_unet" / "metrics_val.json"
DATA_ROOT  = Path(__file__).parent.parent / "data"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    from galaxeye_cd.model import UNetSmall
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = UNetSmall(
        in_channels=int(ckpt.get("in_channels", 6)),
        dropout_p=float(ckpt.get("dropout_p", 0.3)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    threshold = float(ckpt.get("best_threshold", 0.5))
    return model, threshold


@st.cache_data(show_spinner="Building sample list…")
def get_sample_list(split: str):
    from galaxeye_cd.dataset import build_sample_list, discover_split_root
    from galaxeye_cd.config import load_config
    cfg, _ = load_config(Path(__file__).parent / "configs" / "config.yaml")
    sroot = cfg.splits.get(split) or discover_split_root(cfg.data_root, split)
    _, idx_list = build_sample_list(Path(sroot), cfg.folders)
    return idx_list, cfg.img_size


def run_inference(model, img_tensor, use_tta: bool):
    x = img_tensor.unsqueeze(0)
    with torch.no_grad():
        if use_tta:
            from galaxeye_cd.tta import tta_predict
            prob = tta_predict(model, x)
        else:
            prob = torch.sigmoid(model(x))
    return prob.squeeze().numpy()


def norm(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return ((a - lo) / (hi - lo + 1e-9)).clip(0, 1)


# ── Sidebar ───────────────────────────────────────────────────────────────────

hist  = load_json(HIST_PATH) or []
sweep = load_json(SWEEP_PATH)
val_m = load_json(VAL_PATH)

st.sidebar.title("🛰️ GalaxEye CD")
st.sidebar.markdown(f"**Epochs trained:** {len(hist)}")
if sweep:
    st.sidebar.markdown(f"**Best threshold:** `{sweep['best_threshold']:.3f}`")
    st.sidebar.markdown(f"**Best val F1:** `{sweep['best_f1']:.4f}`")
if not CKPT_PATH.exists():
    st.sidebar.error("No checkpoint found. Train first.")
    st.stop()
st.sidebar.success("Checkpoint loaded")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs(["📋 Overview", "📈 Training Curves", "🎚 Threshold Sweep", "🔍 Test Inference", "📤 Upload & Predict"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Run: `baseline_unet`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs", len(hist))
    c2.metric("Best val F1", f"{sweep['best_f1']:.4f}" if sweep else (f"{hist[-1]['val_f1']:.4f}" if hist else "—"))
    c3.metric("Best val IoU", f"{max(h['val_iou'] for h in hist):.4f}" if hist else "—")
    thr = sweep["best_threshold"] if sweep else 0.5
    c4.metric("Decision threshold", f"{thr:.3f}", delta=f"{thr - 0.5:+.3f} vs 0.5")

    if val_m and val_m.get("metrics"):
        st.markdown("---")
        st.subheader("Validation metrics (full set, TTA on)")
        m = val_m["metrics"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("IoU",       f"{m['iou']:.4f}")
        d2.metric("F1",        f"{m['f1']:.4f}")
        d3.metric("Precision", f"{m['precision']:.4f}")
        d4.metric("Recall",    f"{m['recall']:.4f}")

    st.markdown("---")
    st.subheader("Config")
    cfg_file = Path(__file__).parent / "configs" / "config.yaml"
    if cfg_file.exists():
        st.code(cfg_file.read_text(encoding="utf-8"), language="yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Training Curves
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Training curves")
    if not hist:
        st.info("No training history found. Run `train.py` first.")
    else:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            epochs = [h["epoch"] for h in hist]
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=["Train Loss", "Val F1", "Val IoU",
                                "Learning Rate", "Val Precision", "Val Recall"],
            )
            fig.add_trace(go.Scatter(x=epochs, y=[h["train_loss"]    for h in hist], name="Loss",      line=dict(color="#EF553B")), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=[h["val_f1"]        for h in hist], name="F1",        line=dict(color="#00CC96")), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=[h["val_iou"]       for h in hist], name="IoU",       line=dict(color="#636EFA")), row=1, col=3)
            if "lr" in hist[0]:
                fig.add_trace(go.Scatter(x=epochs, y=[h["lr"]            for h in hist], name="LR",        line=dict(color="#FECB52")), row=2, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=[h["val_precision"]  for h in hist], name="Precision", line=dict(color="#AB63FA")), row=2, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=[h["val_recall"]     for h in hist], name="Recall",    line=dict(color="#FFA15A")), row=2, col=3)
            fig.update_layout(height=560, template="plotly_dark", legend_title="Metric")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            import pandas as pd
            df = pd.DataFrame(hist).set_index("epoch")
            st.line_chart(df[["train_loss"]])
            st.line_chart(df[["val_f1", "val_iou", "val_precision", "val_recall"]])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Threshold Sweep
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Threshold sweep (val set)")
    if not sweep:
        st.info("No threshold sweep data found.")
    else:
        try:
            import plotly.graph_objects as go
            thrs = [s["threshold"] for s in sweep["sweep"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=thrs, y=[s["f1"]        for s in sweep["sweep"]], name="F1",        line=dict(color="#00CC96", width=3)))
            fig.add_trace(go.Scatter(x=thrs, y=[s["precision"] for s in sweep["sweep"]], name="Precision", line=dict(color="#AB63FA")))
            fig.add_trace(go.Scatter(x=thrs, y=[s["recall"]    for s in sweep["sweep"]], name="Recall",    line=dict(color="#FFA15A")))
            best_thr = sweep["best_threshold"]
            fig.add_vline(x=best_thr, line_dash="dash", line_color="white",
                          annotation_text=f"best={best_thr:.3f}", annotation_position="top right")
            fig.add_vline(x=0.5, line_dash="dot", line_color="gray",
                          annotation_text="default 0.5", annotation_position="top left")
            fig.update_layout(xaxis_title="Threshold", yaxis_title="Score",
                              template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

            e1, e2, e3 = st.columns(3)
            e1.metric("Best threshold", f"{sweep['best_threshold']:.3f}")
            e2.metric("F1 at best thr", f"{sweep['best_f1']:.4f}")
            f1_05 = next((s["f1"] for s in sweep["sweep"] if abs(s["threshold"] - 0.5) < 0.03), None)
            e3.metric("F1 at 0.5", f"{f1_05:.4f}" if f1_05 else "—",
                      delta=f"{sweep['best_f1'] - (f1_05 or sweep['best_f1']):+.4f}")
        except ImportError:
            st.json(sweep)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Test Inference
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Inference on test / val data")

    col_left, col_right = st.columns([1, 3])

    with col_left:
        split     = st.radio("Split", ["test", "val"], horizontal=False)
        use_tta   = st.checkbox("TTA (8-fold)", value=False,
                                help="More accurate but ~8x slower")
        custom_thr = st.slider("Decision threshold", 0.1, 0.9,
                               value=float(sweep["best_threshold"]) if sweep else 0.5,
                               step=0.05)
        run_btn   = st.button("Run inference", type="primary", use_container_width=True)

    with col_right:
        try:
            idx_list, img_size = get_sample_list(split)
            n_total = len(idx_list)
            st.caption(f"{n_total} samples in **{split}** split")

            sample_idx = st.slider("Sample index", 0, max(0, n_total - 1), 0)

            if run_btn or True:   # always show on slider change
                from galaxeye_cd.dataset import ChangeDetectionDataset
                ds = ChangeDetectionDataset([idx_list[sample_idx]], img_size=img_size, with_mask=True)
                sample = ds[0]

                model, auto_thr = load_model()
                prob_np = run_inference(model, sample["image"], use_tta)
                pred    = (prob_np > custom_thr).astype(np.uint8)

                img_np = sample["image"].numpy()          # (C, H, W)
                n_pre  = 3 if img_np.shape[0] >= 4 else 1

                pre_rgb  = norm(img_np[:n_pre].transpose(1, 2, 0))
                post_rgb = norm(img_np[n_pre:n_pre+3].transpose(1, 2, 0)) if img_np.shape[0] >= n_pre + 3 else norm(img_np[n_pre:].mean(0))

                has_gt = "mask" in sample
                n_cols = 5 if has_gt else 3
                vis_cols = st.columns(n_cols)

                vis_cols[0].image(pre_rgb,  caption="Pre-event",  use_column_width=True, clamp=True)
                vis_cols[1].image(post_rgb, caption="Post-event", use_column_width=True, clamp=True)
                vis_cols[2].image(prob_np,  caption="Probability map", use_column_width=True, clamp=True)

                if has_gt:
                    gt = (sample["mask"].squeeze().numpy() > 0.5).astype(np.uint8)

                    # error map: green=TP, red=FP, blue=FN
                    H, W = pred.shape
                    err = np.zeros((H, W, 3), dtype=np.float32)
                    err[(pred == 1) & (gt == 1)] = [0, .8, 0]
                    err[(pred == 1) & (gt == 0)] = [.9, 0, 0]
                    err[(pred == 0) & (gt == 1)] = [0, 0, .9]

                    vis_cols[3].image(gt.astype(np.float32),   caption="Ground truth",                       use_column_width=True, clamp=True)
                    vis_cols[4].image(err,                     caption="Error (green=TP, red=FP, blue=FN)",   use_column_width=True, clamp=True)

                    tp = int(((pred == 1) & (gt == 1)).sum())
                    fp = int(((pred == 1) & (gt == 0)).sum())
                    fn = int(((pred == 0) & (gt == 1)).sum())
                    iou  = tp / (tp + fp + fn + 1e-9)
                    f1   = 2*tp / (2*tp + fp + fn + 1e-9)
                    prec = tp / (tp + fp + 1e-9)
                    rec  = tp / (tp + fn + 1e-9)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("IoU",       f"{iou:.4f}")
                    m2.metric("F1",        f"{f1:.4f}")
                    m3.metric("Precision", f"{prec:.4f}")
                    m4.metric("Recall",    f"{rec:.4f}")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Upload & Predict
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Upload your own images")
    st.markdown(
        "Upload a **pre-event** and **post-event** image pair. "
        "The model will predict which areas changed between the two. "
        "Supported formats: PNG, JPG, TIF."
    )

    col_up1, col_up2, col_up3 = st.columns(3)
    with col_up1:
        pre_file  = st.file_uploader("Pre-event image",  type=["png","jpg","jpeg","tif","tiff"], key="pre")
    with col_up2:
        post_file = st.file_uploader("Post-event image", type=["png","jpg","jpeg","tif","tiff"], key="post")
    with col_up3:
        mask_file = st.file_uploader("Ground-truth mask (optional)", type=["png","jpg","jpeg","tif","tiff"], key="mask_up")

    up_thr    = st.slider("Decision threshold", 0.1, 0.9,
                          value=float(sweep["best_threshold"]) if sweep else 0.5,
                          step=0.05, key="up_thr")
    up_tta    = st.checkbox("TTA (8-fold, slower)", value=False, key="up_tta")
    predict_btn = st.button("Predict", type="primary", disabled=(pre_file is None or post_file is None))

    if predict_btn and pre_file and post_file:
        import cv2
        import numpy as np
        from io import BytesIO

        def read_uploaded(f) -> np.ndarray:
            data = np.frombuffer(f.read(), np.uint8)
            img  = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                # Try PIL for TIF support
                from PIL import Image as PILImage
                f.seek(0)
                pil = PILImage.open(f).convert("RGB")
                img = np.array(pil)[:, :, ::-1]   # RGB -> BGR
            if img.ndim == 2:
                img = img[:, :, None]
            return img.astype(np.float32)

        def prep_img(arr: np.ndarray, size: int) -> np.ndarray:
            resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
            if resized.ndim == 2:
                resized = resized[:, :, None]
            mx = float(resized.max()) if resized.max() > 0 else 1.0
            normed = (resized / mx).astype(np.float32)
            return np.transpose(normed, (2, 0, 1))   # (C, H, W)

        with st.spinner("Running model…"):
            model_up, auto_thr_up = load_model()
            ckpt_up = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
            in_ch   = int(ckpt_up.get("in_channels", 6))
            img_sz  = 256

            pre_arr  = read_uploaded(pre_file)
            post_arr = read_uploaded(post_file)

            pre_chw  = prep_img(pre_arr,  img_sz)   # (C1, H, W)
            post_chw = prep_img(post_arr, img_sz)   # (C2, H, W)

            # Match channel count to what the model expects
            # pre + post channels must equal in_ch; split evenly
            ch_each = in_ch // 2
            def pad_or_trim(chw, n):
                c = chw.shape[0]
                if c >= n:
                    return chw[:n]
                return np.concatenate([chw] * (n // c + 1), axis=0)[:n]

            pre_chw  = pad_or_trim(pre_chw,  ch_each)
            post_chw = pad_or_trim(post_chw, ch_each)

            x = np.concatenate([pre_chw, post_chw], axis=0)   # (in_ch, H, W)
            x_tensor = torch.from_numpy(x).float()

            prob_np = run_inference(model_up, x_tensor, up_tta)
            pred    = (prob_np > up_thr).astype(np.uint8)

        st.success("Done!")

        # Display
        n_cols = 4 if mask_file else 3
        disp_cols = st.columns(n_cols)

        # Show pre as RGB (take first 3 channels, or grayscale)
        pre_disp = norm(pre_chw[:3].transpose(1, 2, 0)) if pre_chw.shape[0] >= 3 else norm(pre_chw[0])
        post_disp = norm(post_chw[:3].transpose(1, 2, 0)) if post_chw.shape[0] >= 3 else norm(post_chw[0])

        disp_cols[0].image(pre_disp,  caption="Pre-event",       use_column_width=True, clamp=True)
        disp_cols[1].image(post_disp, caption="Post-event",       use_column_width=True, clamp=True)
        disp_cols[2].image(prob_np,   caption=f"Change probability (thr={up_thr:.2f})", use_column_width=True, clamp=True)

        if mask_file:
            mask_arr = read_uploaded(mask_file)
            mask_resized = cv2.resize(mask_arr[:, :, 0] if mask_arr.ndim == 3 else mask_arr,
                                      (img_sz, img_sz), interpolation=cv2.INTER_NEAREST)
            gt = (mask_resized > 0).astype(np.uint8)

            H, W = pred.shape
            err = np.zeros((H, W, 3), dtype=np.float32)
            err[(pred == 1) & (gt == 1)] = [0, .8, 0]
            err[(pred == 1) & (gt == 0)] = [.9, 0, 0]
            err[(pred == 0) & (gt == 1)] = [0, 0, .9]
            disp_cols[3].image(err, caption="Error (green=TP, red=FP, blue=FN)", use_column_width=True, clamp=True)

            tp   = int(((pred==1)&(gt==1)).sum())
            fp   = int(((pred==1)&(gt==0)).sum())
            fn   = int(((pred==0)&(gt==1)).sum())
            iou  = tp/(tp+fp+fn+1e-9)
            f1   = 2*tp/(2*tp+fp+fn+1e-9)
            prec = tp/(tp+fp+1e-9)
            rec  = tp/(tp+fn+1e-9)
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("IoU",       f"{iou:.4f}")
            s2.metric("F1",        f"{f1:.4f}")
            s3.metric("Precision", f"{prec:.4f}")
            s4.metric("Recall",    f"{rec:.4f}")

        # Download prediction
        import io
        from PIL import Image as PILImage
        pred_img = PILImage.fromarray((pred * 255).astype(np.uint8))
        buf = io.BytesIO()
        pred_img.save(buf, format="PNG")
        st.download_button(
            label="Download prediction mask (PNG)",
            data=buf.getvalue(),
            file_name="change_prediction.png",
            mime="image/png",
        )
