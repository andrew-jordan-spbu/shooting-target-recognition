"""
Shooting target auto-detection.

Pipeline
--------
1. Resize frame to a working resolution (speed).
2. Find the white paper sheet -> mask that excludes the background.
3. Within the paper mask, locate the target centre:
   - primary: weighted average of HoughCircles centres (all rings share one)
   - fallback: darkest compact blob
4. Build a radial brightness profile (MEDIAN per ring -- robust to text).
5. Find where rings end and white border begins, clip profile.
6. Detect ring boundaries as contrast transitions inside the target.
7. Derive 10 scoring zones from the detected boundaries.
8. Fall back: constrained HoughCircles -> centred placeholder.
"""

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import CALIBRATION_PATH

# ── Tunable constants ─────────────────────────────────────────────────────
_MAX_DIM = 960  # px -- largest dimension used for detection
_MIN_PAPER_RATIO = 0.05  # paper must cover >= 5 % of frame
_MIN_RING_GAP = 6  # minimum px between adjacent ring boundaries
_MIN_RINGS = 3  # need at least this many detected boundaries
_PROMINENCE = 8  # min grayscale swing to count as a ring edge


# ── Public API ────────────────────────────────────────────────────────────

def detect_target(frame: np.ndarray) -> Optional[dict]:
    """
    Detect a shooting target and return a calibration zone dict.

    Returns
    -------
    dict with keys: cx, cy, outer_radius, zones, frame_shape, auto_detected
    """
    h, w = frame.shape[:2]

    # 1. Downscale for speed
    scale = min(1.0, _MAX_DIM / max(h, w))
    small = (
        cv2.resize(frame, (int(w * scale), int(h * scale)))
        if scale < 1.0 else frame.copy()
    )
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 2. Find paper mask (restrict search to the white sheet)
    paper_mask = _find_paper_mask(gray, sh, sw)

    # 3. Find target centre within the paper
    cx, cy = _find_target_center(gray, paper_mask, sh, sw)

    # 4-5. Use edge-distance histogram for BOTH outer boundary and ring
    # positions.  Edges exist at every ring boundary regardless of contrast
    # (even light-gray zones 1-3 of PNEVMAT have edges).  Brightness-profile
    # methods wrongly classify light outer rings as "white paper".
    max_r = _paper_radius(cx, cy, sh, sw, paper_mask)
    outer_r, boundaries = _analyze_target_edges(
        gray, paper_mask, cx, cy, max_r
    )

    # Fallback: brightness profile if edge analysis yields too little
    if len(boundaries) < _MIN_RINGS or outer_r < max_r * 0.05:
        profile = _radial_profile(gray, cx, cy, max_r)
        outer_r_br = _target_edge(profile)
        if len(boundaries) < _MIN_RINGS:
            boundaries = _ring_boundaries(profile[:outer_r_br])
        if outer_r < max_r * 0.05:
            outer_r = outer_r_br

    # 6. Build zones
    auto_detected = len(boundaries) >= _MIN_RINGS
    if auto_detected:
        zones_s = _zones_from_boundaries(cx, cy, boundaries, outer_r)
    else:
        hough = _hough_fallback(gray, paper_mask, sh, sw)
        if hough:
            cx, cy, outer_r = hough
            auto_detected = True
        zones_s = _build_zones(cx, cy, outer_r)

    # 6. Scale back to original frame coordinates (circles)
    inv = 1.0 / scale
    zones = [
        {**z,
         "cx": int(z["cx"] * inv),
         "cy": int(z["cy"] * inv),
         "radius": max(1, int(z["radius"] * inv))}
        for z in zones_s
    ]

    return {
        "cx": int(cx * inv),
        "cy": int(cy * inv),
        "outer_radius": max(1, int(outer_r * inv)),
        "zones": zones,
        "frame_shape": [h, w],
        "auto_detected": auto_detected,
    }


# ── Step 2: paper mask ────────────────────────────────────────────────────

def _find_paper_mask(gray: np.ndarray, sh: int, sw: int) -> np.ndarray:
    """
    Return a binary mask covering the white paper sheet.
    Falls back to a centre-weighted ellipse if no clear paper is found.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    frame_area = sh * sw

    _, otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_d = cv2.dilate(edges, kernel, iterations=2)

    for binary in [otsu, edges_d]:
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
            if 4 <= len(approx) <= 8 and area < frame_area * 0.97:
                candidates.append((area, approx))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            for area, quad in candidates:
                # Validate aspect ratio: paper should be roughly rectangular
                # (not a sliver or a near-square when A4 is ~0.71 ratio)
                rect = cv2.minAreaRect(quad)
                rw, rh = rect[1]
                if rw == 0 or rh == 0:
                    continue
                ratio = min(rw, rh) / max(rw, rh)
                if ratio < 0.35 or ratio > 0.95:
                    continue  # too narrow or too square for a paper sheet
                mask = np.zeros((sh, sw), dtype=np.uint8)
                cv2.fillPoly(mask, [quad], 255)
                return mask

    # Fallback: conservative centre ellipse (35 % each axis)
    mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (sw // 2, sh // 2),
        (int(sw * 0.35), int(sh * 0.38)),
        0, 0, 360, 255, -1,
    )
    return mask


# ── Step 3: target centre ─────────────────────────────────────────────────

def _find_target_center(
    gray: np.ndarray,
    paper_mask: np.ndarray,
    sh: int,
    sw: int,
) -> Tuple[int, int]:
    """
    Locate the shooting target centre within the paper mask.

    Uses gradient vector voting: every strong edge pixel casts a vote
    along its gradient direction.  All concentric rings share one centre,
    so their gradient lines converge there regardless of ring size or
    whether the ring extends beyond the image boundary.
    Fallback: dark-blob centroid.
    """
    cx, cy = _center_gradient_voting(gray, paper_mask, sh, sw)
    if cx is not None:
        return cx, cy
    return _center_from_blob(gray, paper_mask, sh, sw)


def _center_gradient_voting(
    gray: np.ndarray, mask: np.ndarray, sh: int, sw: int
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the centre of concentric rings via gradient vector voting.

    Principle
    ---------
    Every pixel on a circle boundary has a gradient that points radially
    toward (or away from) the circle's centre.  By drawing a line through
    each strong-gradient pixel along its gradient direction, all lines from
    concentric rings converge at the shared centre.  This works even for
    rings that extend beyond the image boundary (partial arcs still vote
    correctly), and is not fooled by small logos or text whose gradient
    lines scatter in different directions.

    Implementation
    --------------
    * Compute Sobel gradients on the paper area.
    * Sub-sample up to 3 000 of the strongest edge pixels (for speed).
    * For each, cast a vote-line via cv2.line into a float accumulator.
    * Gaussian-smooth the accumulator, find its maximum → centre.
    * Return None if the peak is suspiciously weak (< 5 × mean).
    """
    work = gray.copy()
    work[mask == 0] = 128  # neutral so boundary doesn't create fake edges

    blurred = cv2.GaussianBlur(work, (5, 5), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # Build an "active" mask: only vote with gradients that come from DARK
    # pixels (the target body and its ring boundaries).
    # This excludes:
    #   - Logos / text on the white paper border (bright area → excluded)
    #   - Bright windows / specular reflections (> 200 → excluded)
    #   - Medium-bright curtain patterns outside the paper
    # Dilate the dark region slightly so ring-boundary pixels (which sit
    # between a dark and a bright zone) are included.
    dark_region = ((gray < 160) & (mask > 0)).astype(np.uint8) * 255
    dil_k = max(11, min(sh, sw) // 30) | 1
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
    dark_region = cv2.dilate(dark_region, dil_kernel, iterations=2)
    dark_region = cv2.bitwise_and(dark_region, mask)

    mag_in = mag.copy()
    mag_in[dark_region == 0] = 0.0

    valid_pix = mag_in[dark_region > 0]
    if valid_pix.size < 50:
        # No dark region detected at all – fall back to full-mask voting
        mag_in = mag.copy()
        mag_in[mask == 0] = 0.0
        valid_pix = mag_in[mask > 0]

    thr = float(np.percentile(valid_pix, 85))
    ys, xs = np.where(mag_in > thr)
    if ys.size == 0:
        return None, None

    # Sub-sample for speed
    if ys.size > 3000:
        idx = np.random.default_rng(0).choice(ys.size, 3000, replace=False)
        ys, xs = ys[idx], xs[idx]

    accum = np.zeros((sh, sw), dtype=np.float32)
    half = max(sh, sw)  # line half-length in pixels

    for y, x in zip(ys.tolist(), xs.tolist()):
        m = float(mag_in[y, x])
        if m < 1.0:
            continue
        dx = float(gx[y, x]) / m
        dy = float(gy[y, x]) / m
        x1 = int(x - half * dx)
        y1 = int(y - half * dy)
        x2 = int(x + half * dx)
        y2 = int(y + half * dy)
        cv2.line(accum, (x1, y1), (x2, y2), 1.0, 1)

    # Mask and smooth
    accum[mask == 0] = 0.0
    accum = cv2.GaussianBlur(accum, (51, 51), 0)

    _, peak_val, _, peak_loc = cv2.minMaxLoc(accum)
    mean_val = float(accum[mask > 0].mean()) if accum[mask > 0].size else 0.0
    if mean_val > 0 and peak_val < 5.0 * mean_val:
        return None, None  # no clear convergence

    px, py = int(peak_loc[0]), int(peak_loc[1])

    # Sanity: the target centre must NOT be in a bright (white paper / window)
    # region.  If the voting peak lands in a bright pixel, it is almost
    # certainly a false convergence caused by non-target edges.
    if 0 <= py < sh and 0 <= px < sw and int(gray[py, px]) > 190:
        return None, None

    return px, py


def _center_from_blob(
    gray: np.ndarray, mask: np.ndarray, sh: int, sw: int
) -> Tuple[int, int]:
    """
    Fallback: centroid of the largest dark circular blob inside the paper.

    Strategy
    --------
    1. Set outside-mask pixels to white (avoids bleed-in of dark edges).
    2. Otsu-threshold the paper area to separate target body from white paper.
    3. Morphological closing bridges the small gaps between concentric rings
       so the whole target body merges into one connected component.
    4. The largest connected component = target body; its centroid = centre.
    5. If nothing qualifies, fall back to the blurred-minimum location.
    """
    work = gray.copy()
    work[mask == 0] = 255  # background → white, won't confuse dark-finder

    paper_vals = work[mask > 0]
    if paper_vals.size == 0:
        return sw // 2, sh // 2

    # Use the 20th-percentile of paper pixels as threshold, capped at 120.
    # This captures mainly the very-dark target body while excluding the
    # medium-dark printed numbers (which Otsu would include, causing all text
    # to merge into a giant off-centre blob after morphological closing).
    thr = min(int(np.percentile(paper_vals, 20)), 120)
    _, dark = cv2.threshold(work, thr, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, mask)

    # Small close: bridge ring gaps (~5-15 px) without merging distant text.
    # kernel=9, iter=2 → effective gap fill ≈ 18 px.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find connected components; pick most-circular large blob = target body.
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark, connectivity=8
    )
    paper_area = float(np.count_nonzero(mask)) or float(sh * sw)
    best_score, best_cx, best_cy = 0.0, sw // 2, sh // 2

    for i in range(1, len(stats)):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < paper_area * 0.04:
            continue
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        if not (0 <= cy < sh and 0 <= cx < sw and mask[cy, cx] > 0):
            continue
        bw = float(stats[i, cv2.CC_STAT_WIDTH])
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        circ = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
        # Score: large AND circular (target body wins over text patches)
        score = area * circ
        if score > best_score:
            best_score, best_cx, best_cy = score, cx, cy

    if best_score > 0:
        return best_cx, best_cy

    # Final fallback: blurred minimum within paper
    blurred = cv2.GaussianBlur(work, (21, 21), 0)
    _, _, min_loc, _ = cv2.minMaxLoc(blurred)
    return int(min_loc[0]), int(min_loc[1])


# ── Step 4a: max radius from paper ────────────────────────────────────────

def _paper_radius(
    cx: int, cy: int, sh: int, sw: int, mask: np.ndarray
) -> int:
    """Distance from centre to nearest paper edge (distance transform)."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cy_c = min(max(cy, 0), sh - 1)
    cx_c = min(max(cx, 0), sw - 1)
    r = int(dist[cy_c, cx_c])
    if r < 10:
        r = min(cx_c, cy_c, sw - cx_c, sh - cy_c)
    return max(r, 10)


# ── Step 4b: radial brightness profile ───────────────────────────────────

def _radial_profile(
    gray: np.ndarray, cx: int, cy: int, max_r: int
) -> np.ndarray:
    """
    Median pixel brightness at each integer radius from (cx, cy).

    Median is used instead of mean: text labels and logos printed on the
    target appear at isolated angles and do not affect the median, making
    the profile much more robust for ring boundary detection.
    """
    sh, sw = gray.shape
    max_r = min(max_r, sh, sw)

    y_idx, x_idx = np.mgrid[0:sh, 0:sw]
    dist = np.hypot(x_idx - cx, y_idx - cy).astype(np.int32)
    valid = dist < max_r

    dist_v = dist[valid]
    gray_v = gray[valid].astype(np.float32)

    # Sort by distance to enable fast per-radius grouping
    order = np.argsort(dist_v, kind="stable")
    dist_s = dist_v[order]
    gray_s = gray_v[order]

    # Split into per-radius groups and compute median for each
    splits = np.where(np.diff(dist_s))[0] + 1
    groups_d = np.split(dist_s, splits)
    groups_g = np.split(gray_s, splits)

    profile = np.zeros(max_r, dtype=np.float32)
    for gd, gg in zip(groups_d, groups_g):
        r = int(gd[0])
        if r < max_r and gg.size > 0:
            profile[r] = float(np.median(gg))

    # Fill any zero gaps (very small radii can miss integer values)
    nz = np.nonzero(profile)[0]
    if len(nz) > 1:
        filled = np.interp(np.arange(max_r), nz, profile[nz])
        profile[profile == 0] = filled[profile == 0].astype(np.float32)

    return profile


# ── Step 4c: find where target ends and white paper begins ────────────────

def _target_edge(profile: np.ndarray) -> int:
    """
    Find the last radius still inside the target's scored area.

    The white paper border appears as a sustained bright plateau at the
    end of the profile.  Scan inward to find the last dark-enough position.
    """
    n = len(profile)
    if n < 20:
        return n - 1

    k = max(5, n // 20)
    smooth = np.convolve(
        profile, np.ones(k, dtype=np.float32) / k, mode="same"
    ).astype(float)

    outer = smooth[int(n * 0.80):]
    if len(outer) == 0 or float(outer.max()) < 140:
        return n - 1  # no clear white border found
    paper_level = float(np.median(outer))

    inner = smooth[:max(1, int(n * 0.15))]
    dark_level = float(np.median(inner))

    # 68 % of the way from dark to paper = "we are in the white border"
    threshold = dark_level + (paper_level - dark_level) * 0.68

    for r in range(n - 1, n // 8, -1):
        if smooth[r] <= threshold:
            return min(r + k // 2 + 1, n - 1)

    return int(n * 0.65)


# ── Step 4-5 (unified): edge-histogram outer boundary + ring positions ────

def _analyze_target_edges(
    gray: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    max_r: int,
) -> Tuple[int, list]:
    """
    Return (outer_r, boundaries) using a single edge-distance histogram.

    Why edges, not brightness
    -------------------------
    Brightness-profile methods see light outer rings (zones 1-3 on PNEVMAT)
    as "white paper" and cut off the search too early.  Canny edges exist at
    every ring boundary regardless of whether the ring is dark or light-grey;
    the outermost ring with a detectable edge IS the target boundary.

    Algorithm
    ---------
    1. Canny edges inside the paper mask (neutral grey outside so no
       spurious boundary edges are created).
    2. Distance histogram: each edge pixel votes for r = dist(pixel, centre).
    3. Outer boundary = last radius whose smoothed vote count is clearly
       above the blank-paper noise floor (no rings → almost no edges).
    4. Ring boundaries = local maxima of the histogram inside outer_r whose
       peak height exceeds a significance threshold.
    """
    work = gray.copy()
    work[mask == 0] = 128
    edges = cv2.Canny(cv2.GaussianBlur(work, (5, 5), 0), 25, 80)
    edges = cv2.bitwise_and(edges, mask)

    ys, xs = np.where(edges > 0)
    if ys.size == 0:
        return int(max_r * 0.65), []

    dists = np.hypot(xs - cx, ys - cy).astype(np.int32)
    dists = dists[(dists > 2) & (dists < max_r)]
    if dists.size < 20:
        return int(max_r * 0.65), []

    hist = np.bincount(dists, minlength=max_r).astype(np.float32)

    # ── Normalise by actual pixel count at each radius ────────────────────
    # Dividing by theoretical 2πr is wrong: rings with lower contrast
    # (e.g. outer light-grey zones) produce fewer Canny edges per unit arc
    # than high-contrast rings, and rings near the image/mask boundary have
    # fewer visible pixels than 2πr predicts.
    # Better: normalise by how many pixels from the mask actually fall at
    # each integer radius.  This gives "fraction of mask pixels that are
    # edges" — a contrast-independent relative measure that stays valid for
    # partial arcs (clipped rings, angled shots).
    sh_g, sw_g = gray.shape
    y_all, x_all = np.mgrid[0:sh_g, 0:sw_g]
    all_dists = np.hypot(x_all - cx, y_all - cy).astype(np.int32)
    all_dists_masked = all_dists[mask > 0]
    all_dists_masked = all_dists_masked[all_dists_masked < max_r]
    pixel_counts = np.bincount(
        all_dists_masked, minlength=max_r
    ).astype(np.float32)
    # Avoid division by zero at radii with no mask pixels
    hist_norm = hist / np.maximum(pixel_counts, 1.0)

    # Smooth with a kernel ~1.5 % of max_r so adjacent rings stay separate
    k = max(3, max_r // 65) | 1
    smooth = np.convolve(
        hist_norm, np.ones(k, dtype=np.float32) / k, mode="same"
    )

    # Background density = outer 15 % (blank paper → almost no edges)
    bg_slice = smooth[int(max_r * 0.85):]
    bg_level = float(np.median(bg_slice)) if bg_slice.size else 1e-4
    bg_level = max(bg_level, 1e-4)

    # ── Outer boundary ────────────────────────────────────────────────────
    outer_thr = bg_level * 2.5
    outer_r = int(max_r * 0.65)
    for r in range(max_r - 1, max_r // 8, -1):
        if smooth[r] >= outer_thr:
            outer_r = min(r + k * 2, max_r - 1)
            break

    # ── Ring boundaries ───────────────────────────────────────────────────
    inside = smooth[:outer_r]
    sig_thr = float(inside.mean()) + float(inside.std())
    sig_thr = max(sig_thr, bg_level * 2.0)

    min_gap = max(_MIN_RING_GAP, outer_r // 15)
    skip_r = max(2, int(outer_r * 0.03))
    boundaries = []
    last = -min_gap

    for i in range(skip_r, outer_r - 1):
        if i - last < min_gap:
            continue
        if (smooth[i] >= sig_thr
                and smooth[i] >= smooth[i - 1]
                and smooth[i] >= smooth[i + 1]):
            boundaries.append(i)
            last = i

    return outer_r, boundaries


# ── Step 5 (fallback): explicit edge-distance ring detection ──────────────

def _ring_boundaries_from_edges(
    gray: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    outer_r: int,
) -> list:
    """
    Find ring boundary radii using edge-pixel distance voting (RANSAC-like).

    Every Canny edge pixel inside the target area votes for the integer
    radius r = round(dist(pixel, centre)).  Concentric rings concentrate
    many votes at specific radii; scattered logo/text edges spread their
    votes thinly and don't form significant peaks.

    Steps
    -----
    1. Canny edge detection on the masked, blurred image.
    2. For each edge pixel compute its distance to (cx, cy).
    3. Build a histogram of distances 0 … outer_r.
    4. Smooth the histogram; find peaks whose vote count exceeds a
       significance threshold (mean + k*std of the background).
    5. Return the peak radii, ignoring those < 3 % of outer_r (noise
       from imprecise centre) and duplicates closer than min_gap.
    """
    # Compute edges within the paper mask
    work = gray.copy()
    work[mask == 0] = 128  # neutral grey at boundary -- no fake edges
    blurred = cv2.GaussianBlur(work, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_and(edges, mask)

    # Only keep edge pixels within the target area (< outer_r from centre)
    ys, xs = np.where(edges > 0)
    if ys.size == 0:
        return []
    dists = np.hypot(xs - cx, ys - cy)
    dists = dists[dists < outer_r].astype(np.int32)
    if dists.size < 20:
        return []

    # Build distance histogram
    hist = np.bincount(dists, minlength=outer_r).astype(np.float32)

    # Smooth: kernel ~2 % of outer_r so nearby ring pairs stay separate
    k = max(3, outer_r // 50) | 1
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(hist, kernel, mode="same")

    # Significance threshold: mean + 1.5 * std of smoothed histogram
    bg_mean = float(smooth.mean())
    bg_std = float(smooth.std())
    sig_thr = bg_mean + 1.5 * bg_std

    # Find local maxima above threshold
    min_gap = max(_MIN_RING_GAP, outer_r // 15)
    boundaries = []
    last = -min_gap
    skip_r = int(outer_r * 0.03)  # ignore tiny radii near centre

    for i in range(skip_r, outer_r - 1):
        if i - last < min_gap:
            continue
        if (smooth[i] >= sig_thr
                and smooth[i] >= smooth[i - 1]
                and smooth[i] >= smooth[i + 1]):
            boundaries.append(i)
            last = i

    return boundaries


# ── Step 5 (fallback): brightness-profile ring boundaries ─────────────────

def _ring_boundaries(profile: np.ndarray) -> list:
    """
    Find ring boundary radii as local extrema (dark<->light transitions)
    in the smoothed radial profile.
    """
    n = len(profile)
    if n < 20:
        return []

    k = max(5, n // 30)
    smooth = np.convolve(
        profile, np.ones(k, dtype=np.float32) / k, mode="same"
    )
    deriv = np.diff(smooth.astype(np.float64))

    min_gap = max(_MIN_RING_GAP, n // 30)
    boundaries = []
    last = -min_gap

    for i in range(1, len(deriv)):
        if i - last < min_gap:
            continue
        if deriv[i - 1] * deriv[i] < 0:  # sign change = local extremum
            lo = max(0, i - min_gap)
            hi = min(n, i + min_gap)
            prominence = float(smooth[lo:hi].max() - smooth[lo:hi].min())
            if prominence >= _PROMINENCE:
                boundaries.append(i)
                last = i

    return [r for r in boundaries if r > n * 0.03]


# ── Step 5: build zone lists ──────────────────────────────────────────────

def _zones_from_boundaries(
    cx: int, cy: int, boundaries: list, outer_r: int
) -> list:
    """
    Assign 10 scoring zones given detected ring boundaries and outer radius.

    Standard 10-ring shooting targets have equal-width rings, so the
    canonical zone radii are outer_r × i/10 for i = 1..10.

    When ≥ 10 boundaries are detected the detected positions refine the
    spacing.  When fewer are found we fall back to equal spacing — this
    ensures that scaling all zones by one factor (as the UI drag does)
    keeps zone 10 (bullseye) AND zone 1 (outer) simultaneously correct,
    which is impossible if the inner radius is wrong.
    """
    radii = sorted(r for r in boundaries if 0 < r <= outer_r)
    n = len(radii)

    if n >= 10:
        # Enough detected boundaries: sub-sample to 10
        indices = [int(round(i * (n - 1) / 9)) for i in range(10)]
        final_radii = [radii[idx] for idx in indices]
        # Ensure last radius == outer_r
        final_radii[-1] = outer_r
    else:
        # Fewer than 10 detected: assume equal-width rings.
        # Ring width = outer_r / 10, zone i outer edge = outer_r * i / 10.
        ring_w = outer_r / 10
        final_radii = [int(ring_w * (i + 1)) for i in range(10)]

    return [
        {"score": 10 - i, "cx": cx, "cy": cy, "radius": max(1, r)}
        for i, r in enumerate(final_radii)
    ]


def _build_zones(cx: int, cy: int, outer_radius: int) -> list:
    """Fallback: 10 evenly-spaced zones."""
    return [
        {
            "score": s, "cx": cx, "cy": cy,
            "radius": max(1, int(outer_radius * (11 - s) / 10)),
        }
        for s in range(10, 0, -1)
    ]


# ── Hough fallback ────────────────────────────────────────────────────────

def _hough_fallback(
    gray: np.ndarray, mask: np.ndarray, sh: int, sw: int
) -> Optional[Tuple[int, int, int]]:
    """HoughCircles constrained to the paper mask area."""
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(masked, 40, 120)
    min_r = min(sh, sw) // 10
    max_r = min(sh, sw) // 2

    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.5,
        minDist=min(sh, sw) // 3,
        param1=50, param2=20,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    return int(best[0]), int(best[1]), int(best[2])


# ── Semi-auto: find zone boundary at click point ──────────────────────────

def find_zone_at_click(
    frame: np.ndarray, click_x: int, click_y: int
) -> Tuple[int, int, int]:
    """
    Find the ring zone that contains the user's click and return its
    minimum enclosing circle as (cx, cy, radius).

    No prior knowledge of the target centre is needed — the centre
    emerges from the fitted circle.

    Algorithm
    ---------
    1. Canny edges (low thresholds) + morphological closing → ring
       boundaries become solid, closed barriers.
    2. Connected-component analysis on non-edge pixels → each closed
       ring region is one component.
    3. The component that contains the click point is the user's zone.
    4. ``cv2.minEnclosingCircle`` on its contour gives (cx, cy, radius).

    Fallback: if the region is too large (open boundary at image edge)
    the raw distance from the image centre is used as the radius.
    """
    h, w = frame.shape[:2]
    gray = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 else frame.copy()
    )

    # Low Canny thresholds to catch even low-contrast zone boundaries
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 70)

    # Close small gaps so all ring contours form closed barriers
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker, iterations=2)

    walkable = (edges == 0).astype(np.uint8)

    # Clamp seed point to image bounds
    sx = min(max(click_x, 0), w - 1)
    sy = min(max(click_y, 0), h - 1)

    # If click landed on an edge pixel, nudge to nearest walkable pixel
    if not walkable[sy, sx]:
        found = False
        for d in range(1, 25):
            for dy, dx in [(0, d), (0, -d), (d, 0), (-d, 0),
                           (d, d), (-d, -d), (d, -d), (-d, d)]:
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < h and 0 <= nx < w and walkable[ny, nx]:
                    sy, sx = ny, nx
                    found = True
                    break
            if found:
                break

    # Connected components (4-connectivity keeps ring interiors separate)
    _, labels = cv2.connectedComponents(walkable, connectivity=4)
    zone_label = int(labels[sy, sx])

    if zone_label == 0:
        return click_x, click_y, max(10, min(h, w) // 10)

    zone_mask = ((labels == zone_label) * 255).astype(np.uint8)
    zone_area = int(zone_mask.sum() // 255)

    # If zone covers > 85 % of image the barrier was not closed → fallback
    if zone_area > 0.85 * h * w:
        r_fb = int(math.hypot(click_x - w // 2, click_y - h // 2))
        return click_x, click_y, max(10, r_fb)

    contours, _ = cv2.findContours(
        zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return click_x, click_y, max(10, int(math.sqrt(zone_area / math.pi)))

    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        M = cv2.moments(zone_mask)
        if M["m00"] > 0:
            return (
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"]),
                int(math.sqrt(zone_area / math.pi)),
            )
        return click_x, click_y, 50

    (cx_f, cy_f), radius_f = cv2.minEnclosingCircle(c)
    return int(cx_f), int(cy_f), int(radius_f)


# ── Semi-auto helper: snap known radius to nearest ring edge ─────────────

def snap_to_ring_boundary(
    frame: np.ndarray, cx: int, cy: int, r_click: int
) -> int:
    """
    Given a target centre (cx, cy) and the distance r_click from the
    user's click to that centre, return the radius of the nearest actual
    ring boundary.

    Builds a Canny edge-distance histogram from (cx, cy) and finds the
    peak closest to r_click within a search window.
    Falls back to r_click if no peak is found.
    """
    h, w = frame.shape[:2]
    gray = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 else frame.copy()
    )
    max_r = min(h, w)
    search_w = max(20, max_r // 10)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 25, 80)
    ys, xs = np.where(edges > 0)
    if ys.size == 0:
        return r_click

    dists = np.hypot(xs - cx, ys - cy).astype(np.int32)
    dists = dists[(dists > 0) & (dists < max_r)]
    if dists.size == 0:
        return r_click

    hist = np.bincount(dists, minlength=max_r).astype(np.float32)
    smooth = np.convolve(hist, np.ones(7) / 7, mode="same")

    r_lo = max(1, r_click - search_w)
    r_hi = min(max_r - 1, r_click + search_w)
    window = smooth[r_lo:r_hi]
    if window.size == 0 or window.max() == 0:
        return r_click

    return r_lo + int(np.argmax(window))


# ── Scoring & persistence (public API unchanged) ──────────────────────────

def get_score_from_zones(x: float, y: float, zones: list) -> int:
    """
    Score for point (x, y). Zones must be sorted innermost-first (10->1).
    Supports circle zones {cx,cy,radius} and ellipse zones
    {cx,cy,rx,ry,angle} (for future perspective-corrected zones).
    """
    for zone in zones:
        if "rx" in zone and "ry" in zone:
            if _point_in_ellipse(
                x, y,
                zone["cx"], zone["cy"],
                zone["rx"], zone["ry"],
                zone.get("angle", 0.0),
            ):
                return zone["score"]
        elif math.hypot(x - zone["cx"], y - zone["cy"]) <= zone["radius"]:
            return zone["score"]
    return 0


def _point_in_ellipse(
    px: float, py: float,
    cx: float, cy: float,
    rx: float, ry: float,
    angle_deg: float,
) -> bool:
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    dx, dy = px - cx, py - cy
    x_rot = cos_a * dx + sin_a * dy
    y_rot = -sin_a * dx + cos_a * dy
    return (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1.0


def save_zones(data: dict, path: Path = CALIBRATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_zones(path: Path = CALIBRATION_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
