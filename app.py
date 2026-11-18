"""Streamlit app: planetary landing risk assessment from a satellite image.

Run:  streamlit run app.py
Needs (same folder):  landing_pipeline.py  and  models/crater_yolov8.pt
Optional:             models/boulder_yolov8.pt, models/risk_config.json
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import landing_pipeline as lp

st.set_page_config(page_title="Landing Risk Assessment", page_icon="🌕", layout="wide")

MODELS_DIR = Path("models")
CATEGORY_ICON = {"Safe": "🟢", "Moderate": "🟡", "High": "🟠", "Unsafe": "🔴"}
DECISION_ICON = {"CONTINUE": "✅", "DIVERT": "⚠️", "ABORT": "⛔"}


def fit(fn, *args, **kwargs):
    """Full-width widgets that work on both new and old Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


@st.cache_resource(show_spinner="Loading model...")
def load_yolo(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def decode_image(file, max_side):
    arr = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def png_bytes(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


def sidebar_settings():
    st.sidebar.header("Models")
    crater_path = st.sidebar.text_input("Crater model", str(MODELS_DIR / "crater_yolov8.pt"))
    boulder_default = MODELS_DIR / "boulder_yolov8.pt"
    boulder_path = st.sidebar.text_input("Boulder model (optional)",
                                         str(boulder_default) if boulder_default.exists() else "")

    cfg = lp.load_cfg(MODELS_DIR / "risk_config.json")

    st.sidebar.header("Detection")
    cfg["conf"] = st.sidebar.slider("Confidence threshold", 0.05, 0.90, float(cfg["conf"]), 0.05)
    max_side = st.sidebar.select_slider("Max image side (px)", [640, 800, 1024, 1280, 1600, 2048], 1280)

    st.sidebar.header("Landing site")
    cfg["lander_frac"] = st.sidebar.slider("Lander footprint (% of image side)", 4, 30,
                                           int(cfg["lander_frac"] * 100)) / 100
    cfg["top_k"] = st.sidebar.slider("Number of candidate sites", 1, 10, int(cfg["top_k"]))
    cfg["grid"] = st.sidebar.slider("Risk grid size", 4, 16, int(cfg["grid"]))

    with st.sidebar.expander("Risk weights"):
        for k in list(cfg["weights"]):
            cfg["weights"][k] = st.slider(k.capitalize(), 0.0, 1.0, float(cfg["weights"][k]), 0.05, key=f"w_{k}")
        st.caption("Weights are normalised automatically. Boulder weight is ignored without a boulder model.")

    with st.sidebar.expander("Decision limits (risk score 0-100)"):
        cfg["go_score"] = st.slider("CONTINUE if best site risk <=", 5, 60, int(cfg["go_score"]))
        cfg["caution_score"] = st.slider("DIVERT if best site risk <=", 20, 90, int(cfg["caution_score"]))
    return crater_path, boulder_path, cfg, max_side


def show_overview(r):
    d = r["decision"]
    box = {"CONTINUE": st.success, "DIVERT": st.warning, "ABORT": st.error}[d]
    box(f"{DECISION_ICON[d]} **{d}**: {r['reason']}")

    s = r["summary"]
    c = st.columns(5)
    c[0].metric("Craters", s["crater_count"])
    c[1].metric("Boulders", s["boulder_count"])
    c[2].metric("Mean crater size", f"{s['mean_crater_diam_px']} px")
    c[3].metric("Shadow cover", f"{s['shadow_fraction'] * 100:.1f}%")
    c[4].metric("Best site risk", "n/a" if s["best_site_risk"] is None else f"{s['best_site_risk']}/100")

    left, right = st.columns(2)
    fit(left.image, r["images"]["Detections (red=crater, orange=boulder)"], caption="Detections")
    fit(right.image, r["images"]["Landing sites (1 = best)"], caption="Hazard heatmap and ranked landing sites (green = best)")


def show_maps(r):
    keys = ["Shadow map", "Roughness map", "Slope proxy map", "Crater density map",
            "Hazard probability map", "Uncertainty map"]
    cols = st.columns(3)
    for i, k in enumerate(keys):
        fit(cols[i % 3].image, r["images"][k], caption=k)
    st.caption("Slope here is a proxy from brightness gradients. A true slope needs a DEM (for example LOLA). "
               "Warm colours mean higher value.")


def show_sites(r):
    if not r["sites"]:
        st.info("No valid landing window found (image may be smaller than the lander footprint).")
        return
    df = pd.DataFrame(r["sites"])
    df.insert(1, "", df["category"].map(CATEGORY_ICON))
    fit(st.dataframe, df, hide_index=True)
    st.caption("Risk score: 0 = safest, 100 = most hazardous. Confidence falls where detections are uncertain or shadows hide terrain.")


def show_zones(r, grid):
    st.subheader("Zone risk grid")
    scores = np.array([z["risk_score"] for z in r["zones"]]).reshape(grid, grid)
    df = pd.DataFrame(scores, index=[f"R{i + 1}" for i in range(grid)], columns=[f"C{j + 1}" for j in range(grid)])
    try:
        fit(st.dataframe, df.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=100).format("{:.0f}"))
    except Exception:
        fit(st.dataframe, df)
    counts = pd.Series([z["category"] for z in r["zones"]]).value_counts().reindex(lp.CATS, fill_value=0)
    st.bar_chart(counts)
    with st.expander("Zone table"):
        fit(st.dataframe, pd.DataFrame(r["zones"]), hide_index=True)


def show_detections(r):
    if r["craters"]:
        df = pd.DataFrame(r["craters"]).round(2).sort_values("diam_px", ascending=False)
        st.write(f"**{len(df)} craters**")
        fit(st.dataframe, df, hide_index=True)
        bins = pd.cut(df["diam_px"], bins=[0, 15, 40, 100, 10000], labels=["<15 px", "15-40 px", "40-100 px", ">100 px"])
        st.bar_chart(bins.value_counts().sort_index())
    else:
        st.info("No craters detected at this confidence threshold.")
    if r["boulders"]:
        st.write(f"**{len(r['boulders'])} boulders**")
        fit(st.dataframe, pd.DataFrame(r["boulders"]).round(2), hide_index=True)


def show_downloads(r):
    c1, c2, c3 = st.columns(3)
    c1.download_button("Annotated image (PNG)", png_bytes(r["images"]["Detections (red=crater, orange=boulder)"]),
                       "detections.png", "image/png")
    c2.download_button("Landing sites map (PNG)", png_bytes(r["images"]["Landing sites (1 = best)"]),
                       "landing_sites.png", "image/png")
    c3.download_button("Hazard map (PNG)", png_bytes(r["images"]["Hazard probability map"]),
                       "hazard_map.png", "image/png")
    report = {"decision": r["decision"], "reason": r["reason"], "summary": r["summary"],
              "sites": r["sites"], "zones": r["zones"], "craters": r["craters"], "boulders": r["boulders"]}
    d1, d2, d3 = st.columns(3)
    d1.download_button("Full report (JSON)", json.dumps(report, indent=2), "landing_report.json", "application/json")
    d2.download_button("Sites (CSV)", pd.DataFrame(r["sites"]).to_csv(index=False), "sites.csv", "text/csv")
    d3.download_button("Craters (CSV)", pd.DataFrame(r["craters"]).to_csv(index=False), "craters.csv", "text/csv")


def main():
    st.title("🌕 Autonomous Planetary Landing Risk Assessment")
    st.write("Upload a satellite image of a planetary surface to detect hazards, map risk and rank landing sites.")

    crater_path, boulder_path, cfg, max_side = sidebar_settings()

    if not Path(crater_path).exists():
        st.error(f"Crater model not found at `{crater_path}`. Run the training notebook and place "
                 "`crater_yolov8.pt` in the `models/` folder.")
        st.stop()

    file = st.file_uploader("Satellite image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"])
    if file is None:
        st.info("Upload an image to begin.")
        st.stop()

    img = decode_image(file, max_side)
    if img is None:
        st.error("Could not read this image file.")
        st.stop()

    crater_model = load_yolo(crater_path)
    boulder_model = load_yolo(boulder_path) if boulder_path and Path(boulder_path).exists() else None

    with st.spinner("Analysing terrain..."):
        r = lp.run_pipeline(img, crater_model, boulder_model, cfg)

    t1, t2, t3, t4, t5, t6 = st.tabs(["Overview", "Hazard maps", "Landing sites", "Zone risk", "Detections", "Download"])
    with t1: show_overview(r)
    with t2: show_maps(r)
    with t3: show_sites(r)
    with t4: show_zones(r, cfg["grid"])
    with t5: show_detections(r)
    with t6: show_downloads(r)


if __name__ == "__main__":
    main()
