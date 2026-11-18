"""Landing risk pipeline: shared by the notebook and the Streamlit app."""
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CFG = {
    "conf": 0.25,
    "imgsz": 640,
    "weights": {"crater": 0.35, "boulder": 0.15, "shadow": 0.20, "rough": 0.15, "slope": 0.15},
    "lander_frac": 0.12,          # lander footprint as a fraction of the shorter image side
    "grid": 8,                    # grid size for the zone risk table
    "top_k": 5,
    "cat_thresholds": [0.25, 0.50, 0.75],   # safe / moderate / high / unsafe
    "go_score": 30, "caution_score": 60,    # decision limits on a 0-100 risk score
}
CATS = ["Safe", "Moderate", "High", "Unsafe"]


def load_cfg(path=None):
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    if path and Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


def _norm(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def detect(model, img_bgr, conf=0.25, imgsz=640):
    """Run a YOLO detector and return a list of boxes."""
    if model is None:
        return []
    res = model.predict(img_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(b.conf[0]),
                    "diam_px": float((x2 - x1 + y2 - y1) / 2)})
    return out


def shadow_mask(gray):
    g = cv2.GaussianBlur(gray, (0, 0), 2)
    thr = max(15.0, 0.45 * float(np.median(g)))
    m = (g < thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    min_area = 0.0005 * gray.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep.astype(np.float32)


def roughness_map(gray, k=15):
    g = cv2.GaussianBlur(gray, (0, 0), 1).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    return _norm(cv2.blur(lap, (k, k)))


def slope_proxy(gray, sigma=3):
    """Brightness-gradient proxy for slope (no DEM available from a single image)."""
    g = cv2.GaussianBlur(gray, (0, 0), sigma).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return _norm(np.sqrt(gx ** 2 + gy ** 2))


def coverage_map(shape, boxes, blur=5):
    m = np.zeros(shape, np.float32)
    h, w = shape
    for b in boxes:
        x1, y1 = max(int(b["x1"]), 0), max(int(b["y1"]), 0)
        x2, y2 = min(int(b["x2"]), w), min(int(b["y2"]), h)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = np.maximum(m[y1:y2, x1:x2], b["conf"])
    if blur:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return np.clip(m * 1.5, 0, 1)


def combine(layers, weights):
    avail = {k: w for k, w in weights.items() if k in layers}
    tot = sum(avail.values())
    h = sum(layers[k] * (w / tot) for k, w in avail.items())
    return np.clip(h, 0, 1).astype(np.float32)


def category(score01, thr):
    return CATS[int(np.searchsorted(thr, score01, side="right"))]


def zone_table(hazard, n, thr):
    H, W = hazard.shape
    rows = []
    for i in range(n):
        for j in range(n):
            cell = hazard[i * H // n:(i + 1) * H // n, j * W // n:(j + 1) * W // n]
            s = float(0.6 * cell.mean() + 0.4 * cell.max())
            rows.append({"zone": f"R{i + 1}C{j + 1}", "risk_score": round(100 * s, 1),
                         "category": category(s, thr)})
    return rows


def rank_sites(hazard, uncertainty, lander_px, top_k, thr):
    H, W = hazard.shape
    w = int(max(5, lander_px)) | 1
    mean = cv2.blur(hazard, (w, w))
    mx = cv2.dilate(hazard, np.ones((w, w), np.uint8))
    score = 0.6 * mean + 0.4 * mx
    unc = cv2.blur(uncertainty, (w, w))
    r = w // 2
    valid = np.full(score.shape, np.inf, np.float32)
    if H > 2 * r and W > 2 * r:
        valid[r:H - r, r:W - r] = score[r:H - r, r:W - r]
    sites = []
    for _ in range(top_k):
        idx = np.unravel_index(np.argmin(valid), valid.shape)
        if not np.isfinite(valid[idx]):
            break
        y, x = int(idx[0]), int(idx[1])
        s = float(valid[idx])
        sites.append({"rank": len(sites) + 1, "x": x, "y": y, "radius_px": r,
                      "risk_score": round(100 * s, 1), "category": category(s, thr),
                      "confidence": round(float(1 - unc[y, x]), 2)})
        valid[max(0, y - w):y + w, max(0, x - w):x + w] = np.inf
    return sites


def decision(best, cfg):
    if best is None:
        return "ABORT", "No valid landing window found"
    s = best["risk_score"]
    if s <= cfg["go_score"]:
        return "CONTINUE", "Best site is within the safe limit"
    if s <= cfg["caution_score"]:
        return "DIVERT", "Best site is acceptable only with caution; retarget to the top-ranked site"
    return "ABORT", "No site below the acceptable risk limit"


def _heat(gray_u8_bgr, field, alpha=0.5):
    cm = cv2.applyColorMap((field * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(gray_u8_bgr, 1 - alpha, cm, alpha, 0)


def run_pipeline(img_bgr, crater_model, boulder_model=None, cfg=None):
    cfg = cfg or load_cfg()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    craters = detect(crater_model, img_bgr, cfg["conf"], cfg["imgsz"])
    boulders = detect(boulder_model, img_bgr, cfg["conf"], cfg["imgsz"])

    sh, rough, slope = shadow_mask(gray), roughness_map(gray), slope_proxy(gray)
    layers = {"crater": coverage_map((H, W), craters), "shadow": sh, "rough": rough, "slope": slope}
    if boulder_model is not None:
        layers["boulder"] = coverage_map((H, W), boulders)
    hazard = combine(layers, cfg["weights"])

    low = [b for b in craters + boulders if b["conf"] < 0.5]
    unc = np.clip(coverage_map((H, W), low, blur=7) + 0.5 * cv2.GaussianBlur(sh, (0, 0), 5), 0, 1)

    thr = cfg["cat_thresholds"]
    lander_px = cfg["lander_frac"] * min(H, W)
    sites = rank_sites(hazard, unc, lander_px, cfg["top_k"], thr)
    zones = zone_table(hazard, cfg["grid"], thr)
    verdict, reason = decision(sites[0] if sites else None, cfg)

    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ann = img_bgr.copy()
    for b in craters:
        cv2.rectangle(ann, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 0, 255), 2)
    for b in boulders:
        cv2.rectangle(ann, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 165, 255), 2)
    sites_img = _heat(base, hazard, 0.45)
    for s in sites:
        col = (0, 255, 0) if s["rank"] == 1 else (255, 255, 0)
        cv2.circle(sites_img, (s["x"], s["y"]), s["radius_px"], col, 2)
        cv2.putText(sites_img, str(s["rank"]), (s["x"] - 5, s["y"] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    to_rgb = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    return {
        "images": {
            "Original": to_rgb(img_bgr),
            "Detections (red=crater, orange=boulder)": to_rgb(ann),
            "Shadow map": to_rgb(_heat(base, sh, 0.55)),
            "Roughness map": to_rgb(_heat(base, rough, 0.55)),
            "Slope proxy map": to_rgb(_heat(base, slope, 0.55)),
            "Crater density map": to_rgb(_heat(base, layers["crater"], 0.55)),
            "Hazard probability map": to_rgb(_heat(base, hazard, 0.55)),
            "Uncertainty map": to_rgb(_heat(base, unc, 0.55)),
            "Landing sites (1 = best)": to_rgb(sites_img),
        },
        "craters": craters, "boulders": boulders, "sites": sites, "zones": zones,
        "decision": verdict, "reason": reason,
        "summary": {
            "crater_count": len(craters), "boulder_count": len(boulders),
            "mean_crater_diam_px": round(float(np.mean([c["diam_px"] for c in craters])), 1) if craters else 0.0,
            "shadow_fraction": round(float(sh.mean()), 3),
            "mean_hazard": round(float(hazard.mean()), 3),
            "best_site_risk": sites[0]["risk_score"] if sites else None,
        },
    }
