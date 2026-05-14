"""
Shooting target auto-detection.

Pipeline
--------
1. Resize frame to a working resolution (speed).
2. Find the white paper sheet → quadrilateral mask.
3. Perspective-warp the paper to a rectangle (removes tilt / trapezoid
   distortion) so rings appear as proper circles.
4. Within the warped rectangle, locate the target centre (darkest blob).
5. Build a radial brightness profile from that centre; detect ring
   boundaries as contrast transitions (local extrema).
6. Derive 10 scoring zones from the detected boundaries.
7. Map zone centres and radii back to the original frame coordinates
   via the inverse perspective transform.
8. Fall back through: constrained HoughCircles → centred placeholder.
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import CALIBRATION_PATH

# ── Tunable constants ──────────────────────────────────────────────────────────
_MAX_DIM = 960          # px – largest dimension used for detection
_MIN_PAPER_RATIO = 0.05 # paper must cover at least 5 % of frame area
_MIN_RING_GAP = 6       # minimum px between adjacent ring boundaries
_MIN_RINGS = 3          # need at least this many boundaries for radial method
_PROMINENCE = 6         # min grayscale swing to count as a ring edge


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_target(frame: np.ndarray) -> Optional[dict]:
    """
    Detect a shooting target and return a calibration zone dict.

    Returns
    -------
    dict with keys: cx, cy, outer_radius, zones, frame_shape, auto_detected
    """
    h, w = frame.shape[:2]

    # 1. Downscale for speed; track the scale factor
    scale = min(1.0, _MAX_DIM / max(h, w))
    small = (
        cv2.resize(frame, (int(w * scale), int(h * scale)))
        if scale < 1.0 else frame.copy()
    )
    sh, sw = small.shape[:2]
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 2. Find the paper sheet → mask + optional warp matrix
    paper_mask, M_warp, warp_size = _find_paper(gray_small, sh, sw)

    # 3. Work in warped (perspective-corrected) space when available
    if M_warp is not None:
        gray_work = cv2.warpPerspective(gray_small, M_warp, warp_size)
        wh, ww = gray_work.shape[:2]
        work_mask = np.ones((wh, ww), dtype=np.uint8) * 255
        M_inv = cv2.invert(M_warp)[1]
    else:
        gray_work = gray_small
        wh, ww = sh, sw
        work_mask = paper_mask
        M_inv = None

    # 4. Target centre = darkest blob within working image
    cx_w, cy_w = _find_center(gray_work, work_mask)

    # 5. Radial profile → ring boundaries
    max_r = _paper_radius(cx_w, cy_w, wh, ww, work_mask)
    profile = _radial_profile(gray_work, cx_w, cy_w, max_r)
    boundaries = _ring_boundaries(profile)

    # 6. Build zones
    auto_detected = len(boundaries) >= _MIN_RINGS
    if auto_detected:
        zones_w = _zones_from_boundaries(cx_w, cy_w, boundaries)
        outer_r_w = boundaries[-1]
    else:
        hough = _hough_fallback(gray_work, work_mask, wh, ww)
        if hough:
            cx_w, cy_w, outer_r_w = hough
            auto_detected = True
        else:
            outer_r_w = max_r
        zones_w = _build_zones(cx_w, cy_w, outer_r_w)

    # 7. Map back to original frame coordinates
    if M_inv is not None:
        zones = _warp_zones_back(zones_w, M_inv, scale)
        pt = _transform_point(M_inv, cx_w, cy_w)
        cx_orig = int(pt[0] / scale)
        cy_orig = int(pt[1] / scale)
        outer_radius = _warp_radius_back(M_inv, cx_w, cy_w, outer_r_w, scale)
    else:
        inv = 1.0 / scale
        zones = [
            {**z,
             "cx": int(z["cx"] * inv),
             "cy": int(z["cy"] * inv),
             "radius": max(1, int(z["radius"] * inv))}
            for z in zones_w
        ]
        cx_orig = int(cx_w / scale)
        cy_orig = int(cy_w / scale)
        outer_radius = max(1, int(outer_r_w / scale))

    return {
        "cx": cx_orig,
        "cy": cy_orig,
        "outer_radius": outer_radius,
        "zones": zones,
        "frame_shape": [h, w],
        "auto_detected": auto_detected,
    }


# ── Step 2: find paper ─────────────────────────────────────────────────────────

def _find_paper(
    gray: np.ndarray, sh: int, sw: int
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int]]:
    """
    Locate the white sheet of paper.

    Returns
    -------
    paper_mask  : uint8 mask – 255 where paper is, 0 elsewhere
    M_warp      : 3×3 perspective matrix to warp paper → rectangle, or None
    warp_size   : (width, height) of the warped rectangle
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    frame_area = sh * sw

    # Try two binary images: Otsu-threshold and Canny edges
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    full_mask = np.ones((sh, sw), dtype=np.uint8) * 255

    for binary in [otsu, edges]:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < frame_area * _MIN_PAPER_RATIO:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if 4 <= len(approx) <= 8:
                candidates.append((area, approx))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, quad = candidates[0]

        if len(quad) == 4:
            mask = np.zeros((sh, sw), dtype=np.uint8)
            cv2.fillPoly(mask, [quad], 255)

            M, wsize = _perspective_matrix(quad, sh, sw)
            return mask, M, wsize

        # Non-quad polygon → just use as mask, no warp
        mask = np.zeros((sh, sw), dtype=np.uint8)
        cv2.fillPoly(mask, [quad], 255)
        return mask, None, (sw, sh)

    return full_mask, None, (sw, sh)


def _perspective_matrix(
    quad: np.ndarray, sh: int, sw: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Compute perspective warp from an arbitrary quad to an upright rectangle."""
    pts = quad.reshape(4, 2).astype(np.float32)

    # Order: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype=np.float32)

    tl, tr, br, bl = ordered
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width = max(width, 1)
    height = max(height, 1)

    dst = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(ordered, dst)
    return M, (width, height)


# ── Step 4: locate target centre ──────────────────────────────────────────────

def _find_center(gray: np.ndarray, mask: np.ndarray) -> Tuple[int, int]:
    """
    Return the centre of the target (darkest blob inside the mask).
    Blurs heavily so fine ring texture doesn't dominate.
    """
    sh, sw = gray.shape
    inv = cv2.bitwise_not(gray)
    inv = cv2.bitwise_and(inv, inv, mask=mask)

    blur_k = max(21, (min(sh, sw) // 8) | 1)  # odd number
    blurred = cv2.GaussianBlur(inv, (blur_k, blur_k), 0)

    # Coarse location via global maximum
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx, cy = max_loc

    # Refine with centroid of bright (= dark in original) region
    thr = int(np.percentile(blurred[blurred > 0], 90)) if blurred.max() > 0 else 128
    _, region = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)
    m = cv2.moments(region)
    if m["m00"] > 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

    return cx, cy


# ── Step 5a: estimate usable radius ───────────────────────────────────────────

def _paper_radius(
    cx: int, cy: int, sh: int, sw: int, mask: np.ndarray
) -> int:
    """
    Maximum radius for the radial profile: the distance from the centre to
    the nearest paper edge, obtained via OpenCV distance transform.
    """
    # Distance transform gives distance to nearest zero pixel
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Value at centre = distance to paper boundary
    cy_c = min(max(cy, 0), sh - 1)
    cx_c = min(max(cx, 0), sw - 1)
    r = int(dist[cy_c, cx_c])
    if r < 10:
        r = min(cx_c, cy_c, sw - cx_c, sh - cy_c)
    return max(r, 10)


# ── Step 5b: radial brightness profile ────────────────────────────────────────

def _radial_profile(
    gray: np.ndarray, cx: int, cy: int, max_r: int
) -> np.ndarray:
    """
    Compute mean pixel brightness at each integer radius from (cx, cy).
    Vectorised – runs in < 50 ms even at 960 px.
    """
    sh, sw = gray.shape
    max_r = min(max_r, sh, sw)

    y_idx, x_idx = np.mgrid[0:sh, 0:sw]
    dist = np.hypot(x_idx - cx, y_idx - cy).astype(np.int32)
    mask = dist < max_r

    profile = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.int32)
    np.add.at(profile, dist[mask], gray[mask].astype(np.float64))
    np.add.at(counts, dist[mask], 1)

    valid = counts > 0
    profile[valid] /= counts[valid]
    return profile.astype(np.float32)


# ── Step 5c: detect ring boundaries ───────────────────────────────────────────

def _ring_boundaries(profile: np.ndarray) -> list:
    """
    Find ring boundary radii as local extrema (contrast transitions) in the
    smoothed radial profile.

    Each dark↔light transition in the target corresponds to a sign change in
    the first derivative → local minimum or maximum in the profile.
    """
    n = len(profile)
    if n < 20:
        return []

    # Smooth to suppress fine texture noise
    k = max(5, n // 30)
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(profile, kernel, mode="same")

    deriv = np.diff(smooth.astype(np.float64))

    min_gap = max(_MIN_RING_GAP, n // 30)
    boundaries = []
    last = -min_gap

    for i in range(1, len(deriv)):
        if i - last < min_gap:
            continue
        # Sign change in derivative = local extremum
        if deriv[i - 1] * deriv[i] < 0:
            lo = max(0, i - min_gap)
            hi = min(n, i + min_gap)
            prominence = float(smooth[lo:hi].max() - smooth[lo:hi].min())
            if prominence >= _PROMINENCE:
                boundaries.append(i)
                last = i

    # Ignore boundaries very close to the centre (< 3 % of profile)
    boundaries = [r for r in boundaries if r > n * 0.03]
    return boundaries


# ── Step 6: build zone lists ───────────────────────────────────────────────────

def _zones_from_boundaries(cx: int, cy: int, boundaries: list) -> list:
    """
    Assign 10 scoring zones to the detected ring boundaries.
    innermost boundary → score 10,  outermost → score 1.
    If more than 10 boundaries: sub-sample evenly.
    If fewer than 10: interpolate linearly between inner and outer.
    """
    radii = sorted(boundaries)
    n = len(radii)

    if n >= 10:
        # Pick 10 evenly-spaced boundaries
        indices = [int(round(i * (n - 1) / 9)) for i in range(10)]
        radii = [radii[i] for i in indices]
    else:
        # Interpolate to 10 evenly-spaced radii
        r_inner = radii[0]
        r_outer = radii[-1]
        step = (r_outer - r_inner) / 9
        radii = [int(r_inner + i * step) for i in range(10)]

    zones = []
    for i, r in enumerate(radii):
        zones.append({
            "score": 10 - i,
            "cx": cx,
            "cy": cy,
            "radius": max(1, int(r)),
        })
    return zones


def _build_zones(cx: int, cy: int, outer_radius: int) -> list:
    """Build 10 evenly-spaced zones (fallback, no real detection)."""
    zones = []
    for score in range(10, 0, -1):
        r = int(outer_radius * (11 - score) / 10)
        zones.append({"score": score, "cx": cx, "cy": cy, "radius": max(1, r)})
    return zones


# ── Step 6b: HoughCircles fallback ────────────────────────────────────────────

def _hough_fallback(
    gray: np.ndarray, mask: np.ndarray, sh: int, sw: int
) -> Optional[Tuple[int, int, int]]:
    """Constrained HoughCircles inside the paper mask using Canny edges."""
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(masked, 40, 120)

    min_r = min(sh, sw) // 10
    max_r = min(sh, sw) // 2

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min(sh, sw) // 3,
        param1=50,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    return int(best[0]), int(best[1]), int(best[2])


# ── Step 7: map results back to original frame ─────────────────────────────────

def _transform_point(
    M: np.ndarray, x: float, y: float
) -> Tuple[float, float]:
    """Apply a 3×3 perspective matrix to a single point."""
    pt = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def _warp_radius_back(
    M_inv: np.ndarray, cx_w: float, cy_w: float, r_w: float, scale: float
) -> int:
    """
    Approximate the radius in original-frame pixels.
    Samples the inverse transform at (cx+r, cy) and (cx, cy+r) and averages.
    """
    cx_o, cy_o = _transform_point(M_inv, cx_w, cy_w)
    rx_o, _ = _transform_point(M_inv, cx_w + r_w, cy_w)
    _, ry_o = _transform_point(M_inv, cx_w, cy_w + r_w)
    r_orig = (abs(rx_o - cx_o) + abs(ry_o - cy_o)) / 2
    return max(1, int(r_orig / scale))


def _warp_zones_back(
    zones: list, M_inv: np.ndarray, scale: float
) -> list:
    """Map all zone centres and radii back to original frame coordinates."""
    result = []
    for z in zones:
        cx_o, cy_o = _transform_point(M_inv, z["cx"], z["cy"])
        r_o = _warp_radius_back(M_inv, z["cx"], z["cy"], z["radius"], scale)
        result.append({
            "score": z["score"],
            "cx": int(cx_o / scale),
            "cy": int(cy_o / scale),
            "radius": r_o,
        })
    return result


# ── Scoring & persistence (unchanged API) ─────────────────────────────────────

def get_score_from_zones(x: float, y: float, zones: list) -> int:
    """
    Score for point (x, y). Zones must be sorted innermost-first (10 → 1).
    """
    for zone in zones:
        if math.hypot(x - zone["cx"], y - zone["cy"]) <= zone["radius"]:
            return zone["score"]
    return 0


def save_zones(data: dict, path: Path = CALIBRATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_zones(path: Path = CALIBRATION_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
