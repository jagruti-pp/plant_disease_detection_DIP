"""
dip_pipeline.py
---------------
Digital Image Processing (DIP) stages for the Plant Leaf Disease Detection
mini project. Shared by app.py (website) and train_model.py (training) so the
classifier sees exactly the same pre-processed image during training and
prediction.

Stages implemented here:
    Acquisition -> Enhancement -> Filtering -> Edge Detection ->
    Thresholding -> Morphology -> (lesion localisation for the final image)

NOTE: The classifier (Keras model) lives in app.py / train_model.py.
      Segmentation here only LOCATES suspicious regions; it never decides
      the disease name.
"""
import cv2
import numpy as np

MAX_SIDE = 1024          # images larger than this are down-scaled for speed
MODEL_SIZE = 224         # default classifier input size


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def norm(x):
    """Clip to 0-255 and convert to uint8."""
    return np.clip(x, 0, 255).astype(np.uint8)


def decode_image(data: bytes):
    """
    Decode raw image bytes into a BGR uint8 image.
    Returns (bgr_image, original_channels).
    Raises ValueError for corrupted / unreadable data.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise ValueError("The file could not be decoded as an image (it may be corrupted).")

    if img.dtype != np.uint8:                       # 16-bit PNG etc.
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    channels = 1 if img.ndim == 2 else img.shape[2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                         # composite alpha on white
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    elif img.shape[2] != 3:
        raise ValueError("Unsupported number of image channels.")
    return img, channels


def limit_size(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = max(h, w) / float(max_side)
    if scale <= 1:
        return img
    return cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Pre-processing used by the classifier (enhancement + filtering)
# --------------------------------------------------------------------------
def classifier_input_bgr(img):
    """Contrast stretching (enhancement) followed by Gaussian filtering."""
    stretched = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(stretched, (3, 3), 0)


def preprocess_for_model(img_bgr, size=MODEL_SIZE):
    """
    BGR uint8 image -> float32 RGB array (size, size, 3) with values 0..255.
    (Rescaling to the network's range happens inside the Keras model.)
    """
    processed = classifier_input_bgr(img_bgr)
    processed = cv2.resize(processed, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32)


# --------------------------------------------------------------------------
# Leaf & lesion localisation (DIP only)
# --------------------------------------------------------------------------
def segment_leaf(img):
    """Return (leaf_mask uint8 0/255, found_bool)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = img.shape[0] * img.shape[1]
    filled = np.zeros_like(mask)
    kept = [c for c in contours if cv2.contourArea(c) > 0.02 * total]
    cv2.drawContours(filled, kept, -1, 255, thickness=cv2.FILLED)   # fills holes

    frac = filled.mean() / 255.0
    if frac < 0.05 or frac > 0.97:        # segmentation unreliable -> whole image
        return np.full(mask.shape, 255, np.uint8), False
    return filled, True


def find_lesions(img, leaf_mask):
    """
    Locate suspicious (non-green / very dark / whitish) pixels inside the leaf.
    Pure colour-based DIP segmentation followed by morphology.
    Returns (lesion_mask uint8 0/255, leaf_pixel_count, lesion_pixel_count).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    side = max(img.shape[:2])
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, side // 100 | 1),) * 2)
    inner = cv2.erode(leaf_mask, ek)                 # ignore the leaf boundary

    inside = inner > 0
    leaf_px = int(np.count_nonzero(leaf_mask))
    if inside.sum() < 50:
        return np.zeros_like(leaf_mask), leaf_px, 0

    median_v = float(np.median(v[inside]))
    not_green = ((h < 33) | (h > 90)) & (s > 40)     # brown / yellow / purple
    dark = v < 0.45 * median_v                       # necrotic / black spots
    whitish = (s < 40) & (v > 190)                   # powdery coating
    lesion = ((not_green | dark | whitish) & inside).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, k3)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, k5)

    # remove tiny specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lesion, connectivity=8)
    min_area = max(20, int(0.001 * leaf_px))
    clean = np.zeros_like(lesion)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[lab == i] = 255
    return clean, leaf_px, int(np.count_nonzero(clean))


def annotate_final(img, label_text, diseased, lesion_mask=None, confidence=None):
    """
    Build the final annotated image.
      diseased=True  -> red translucent overlay + red boundary on lesions
      diseased=False -> leaf shown normally with a green 'Healthy Leaf' banner
      diseased=None  -> leaf shown normally with a grey 'not classified' banner
    """
    out = img.copy()
    if diseased and lesion_mask is not None and lesion_mask.any():
        red = out.copy()
        red[lesion_mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(out, 0.60, red, 0.40, 0)
        cnts, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), max(2, img.shape[1] // 300))

    if diseased:
        color = (40, 40, 200)
    elif diseased is False:
        color = (60, 140, 30)
    else:
        color = (110, 110, 110)

    text = label_text if confidence is None else f"{label_text}  |  {confidence:.1f}%"
    scale = max(0.5, img.shape[1] / 900.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    bar_h = th + base + int(20 * scale)
    bar = np.full((bar_h, out.shape[1], 3), color, np.uint8)
    cv2.putText(bar, text, (int(12 * scale), int(bar_h / 2 + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return np.vstack([out, bar])


# --------------------------------------------------------------------------
# Measurements used to CHOOSE a filter at each step
# --------------------------------------------------------------------------
def noise_sigma(gray):
    """Immerkaer's fast noise-standard-deviation estimate (works on one gray image)."""
    g = gray.astype(np.float32)
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    resp = cv2.filter2D(g, cv2.CV_32F, m)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2.0) / (6.0 * max(w - 2, 1) * max(h - 2, 1)) * np.abs(resp).sum())


def edge_snr(mag):
    """Strong-edge response (mean of top 5 %) divided by flat-area response (mean of lower 50 %)."""
    mag = mag.astype(np.float32)
    top = mag[mag >= np.percentile(mag, 95)].mean()
    low = mag[mag <= np.median(mag)].mean() + 1e-3
    return float(top / low)


def otsu_separability(gray, t):
    """Between-class variance / total variance for threshold t (0..1; high = clearly two-peaked)."""
    g = gray.astype(np.float32)
    var_t = float(g.var())
    if var_t < 1e-6:
        return 0.0
    a, b = g[g <= t], g[g > t]
    if a.size == 0 or b.size == 0:
        return 0.0
    w0, w1 = a.size / g.size, b.size / g.size
    return float(w0 * w1 * (a.mean() - b.mean()) ** 2 / var_t)


def lighting_variation(gray, t):
    """
    How much the brightness of the SAME kind of region (leaf, or background) drifts across the
    image: the image is cut into a 4x4 grid, the mean of each Otsu class is taken per cell, and the
    larger spread of the two classes is returned. Leaf-vs-background contrast does not count.
    """
    h, w = gray.shape
    g = gray.astype(np.float32)
    spreads = []
    for cls in (g <= t, g > t):
        means = []
        for i in range(4):
            for j in range(4):
                cell = cls[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                if cell.sum() > 0.02 * cell.size:
                    means.append(g[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4][cell].mean())
        if len(means) >= 3:
            spreads.append(float(np.std(means)))
    return max(spreads) if spreads else 0.0


def _cand(name, key, selected, reason):
    return {"name": name, "key": key, "selected": bool(selected), "reason": reason}


def _small_components(mask, tiny):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < tiny))


# --------------------------------------------------------------------------
# Full DIP stage computation (Acquisition ... Morphology)
#
# Every step now MEASURES the image it receives, CHOOSES the most suitable
# technique, explains why, and passes the chosen result to the next step:
#   original -> enhancement -> filtering -> edge detection -> thresholding -> morphology
# --------------------------------------------------------------------------
def run_dip_stages(img):
    """
    Returns (images dict name -> uint8 image, numbers dict).
    numbers["decisions"] is a list of 5 dicts (Enhancement, Filtering, Edge Detection,
    Thresholding, Morphology) each holding: stage, input, chosen, output_key, why,
    metrics, candidates[{name,key,selected,reason}].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_px = gray.size
    decisions = []

    # ================= STEP 2 : ENHANCEMENT =================
    p1, p99 = np.percentile(gray, [1, 99])
    span, mean_g, std_g = float(p99 - p1), float(gray.mean()), float(gray.std())
    negative = 255 - img
    contrast = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    gamma_val = float(np.clip(np.log(0.5) / np.log(min(max(mean_g, 1.0), 254.0) / 255.0), 0.5, 2.0))
    lut = np.array([((i / 255.0) ** gamma_val) * 255 for i in range(256)], dtype=np.uint8)
    gamma = cv2.LUT(img, lut)

    if span < 180:
        e_key, e_name = "contrast", "Contrast Stretching"
        e_why = (f"The usable intensity range is only {span:.0f} of 255 levels (1st to 99th percentile), so the image looks flat. "
                 f"Contrast stretching spreads the levels over the full 0-255 range and keeps the leaf colours natural.")
    elif mean_g < 80 or mean_g > 175:
        e_key, e_name = "gamma", f"Gamma Transformation (γ = {gamma_val:.2f})"
        e_why = (f"The range is already wide ({span:.0f} levels) but the mean brightness is {mean_g:.0f}, i.e. the photo is "
                 f"{'too dark' if mean_g < 80 else 'too bright'}. A gamma curve with γ = {gamma_val:.2f} moves the mean towards mid-grey without clipping.")
    else:
        e_key, e_name = "contrast", "Contrast Stretching"
        e_why = (f"Exposure is acceptable (range {span:.0f} levels, mean brightness {mean_g:.0f}), so no brightness correction is needed. "
                 f"A linear contrast stretch is the safest choice: it keeps colour ratios intact, which matters because disease shows up as a colour change.")
    enhanced = contrast if e_key == "contrast" else gamma
    gray_e = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    decisions.append({
        "stage": "Enhancement", "input": "Original image", "chosen": e_name, "output_key": e_key, "why": e_why,
        "metrics": [["Intensity range (p1-p99)", round(span, 1)], ["Mean brightness", round(mean_g, 1)], ["Contrast (std)", round(std_g, 1)]],
        "candidates": [
            _cand("Image Negative", "negative", False, "Not chosen: it inverts every colour, so the green/brown/yellow cues that identify disease would be lost."),
            _cand("Contrast Stretching", "contrast", e_key == "contrast", "Chosen: " + ("range is narrow." if span < 180 else "safe linear stretch, exposure is fine.") if e_key == "contrast" else "Not chosen: the image is not low-contrast and brightness correction is what it needs."),
            _cand(f"Gamma Transformation (γ = {gamma_val:.2f})", "gamma", e_key == "gamma", "Chosen: brightness is skewed." if e_key == "gamma" else "Not chosen: brightness is already acceptable, gamma would only shift it needlessly."),
        ]})

    # ================= STEP 3 : FILTERING =================
    sigma = noise_sigma(gray_e)
    med3 = cv2.medianBlur(gray_e, 3)
    impulse = float((np.abs(gray_e.astype(np.int16) - med3.astype(np.int16)) > 40).mean())
    lap_raw = cv2.Laplacian(gray_e, cv2.CV_64F)
    sharp = float(lap_raw.var())
    lap = cv2.convertScaleAbs(lap_raw)
    gaussian = cv2.GaussianBlur(enhanced, (5, 5), 0)
    median = cv2.medianBlur(enhanced, 5)
    highboost = norm(enhanced.astype(np.float32) + 1.5 * (enhanced.astype(np.float32) - gaussian.astype(np.float32)))

    if impulse > 0.01:
        f_key, f_name = "median", "Median Filter"
        f_why = (f"{impulse*100:.1f}% of pixels are isolated bright/dark outliers (salt-and-pepper type noise). "
                 f"A median filter removes such outliers completely while keeping lesion edges sharp; a Gaussian would only smear them.")
    elif sigma > 4.0:
        f_key, f_name = "gaussian", "Gaussian Filter"
        f_why = (f"The estimated noise level is σ = {sigma:.1f} (above 4) and there are almost no isolated outliers ({impulse*100:.2f}%), "
                 f"so the noise is random/grainy. A Gaussian filter averages it out smoothly.")
    elif sharp < 50:
        f_key, f_name = "highboost", "High-Boost Filtering"
        f_why = (f"The image is clean (noise σ = {sigma:.1f}) but soft: sharpness (Laplacian variance) is only {sharp:.0f}. "
                 f"High-boost filtering restores edge detail and lesion borders without needing any smoothing.")
    else:
        f_key, f_name = "gaussian", "Gaussian Filter"
        f_why = (f"The image is clean (noise σ = {sigma:.1f}) and sharp (Laplacian variance {sharp:.0f}), so only light smoothing is needed. "
                 f"A Gaussian filter suppresses fine leaf texture so the next edge-detection step finds real boundaries, not texture.")
    filtered = {"gaussian": gaussian, "median": median, "highboost": highboost}[f_key]
    gray_f = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    decisions.append({
        "stage": "Filtering", "input": "Enhanced image (" + e_name + ")", "chosen": f_name, "output_key": f_key, "why": f_why,
        "metrics": [["Noise σ", round(sigma, 2)], ["Outlier pixels", f"{impulse*100:.2f}%"], ["Sharpness (Lap. var)", round(sharp, 1)]],
        "candidates": [
            _cand("Gaussian Filter", "gaussian", f_key == "gaussian", "Chosen: best for random/grainy noise." if f_key == "gaussian" else "Not chosen: " + ("outlier noise is better handled by the median." if f_key == "median" else "image is soft, smoothing would blur it more.")),
            _cand("Median Filter", "median", f_key == "median", "Chosen: removes isolated outlier pixels." if f_key == "median" else "Not chosen: there are too few isolated outlier pixels to need it and it rounds off fine lesion detail."),
            _cand("Laplacian", "laplacian", False, "Not chosen as the filter: it is a derivative operator that outputs an edge map and amplifies noise. It is compared again in the Edge Detection step."),
            _cand("High-Boost Filtering", "highboost", f_key == "highboost", "Chosen: image is clean but soft." if f_key == "highboost" else "Not chosen: sharpening would amplify noise/texture" + (" in this image." if f_key != "highboost" and sharp >= 50 else ".")),
        ]})

    # ================= STEP 4 : EDGE DETECTION (on the filtered image) =================
    rx = cv2.filter2D(gray_f, cv2.CV_32F, np.array([[1, 0], [0, -1]], np.float32))
    ry = cv2.filter2D(gray_f, cv2.CV_32F, np.array([[0, 1], [-1, 0]], np.float32))
    roberts_m = cv2.magnitude(rx, ry)
    px = cv2.filter2D(gray_f, cv2.CV_32F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32))
    py = cv2.filter2D(gray_f, cv2.CV_32F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32))
    prewitt_m = cv2.magnitude(px, py)
    sx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    sobel_m = cv2.magnitude(sx, sy)
    lap_m = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F))
    roberts, prewitt, sobel, laplacian_edge = norm(roberts_m), norm(prewitt_m), norm(sobel_m), norm(lap_m)
    snr = {"roberts": edge_snr(roberts_m), "prewitt": edge_snr(prewitt_m), "sobel": edge_snr(sobel_m), "laplacian_edge": edge_snr(lap_m)}
    names = {"roberts": "Roberts", "prewitt": "Prewitt", "sobel": "Sobel", "laplacian_edge": "Laplacian"}
    d_key = max(snr, key=snr.get)
    runner = sorted(snr, key=snr.get, reverse=True)[1]
    d_why = (f"Each operator was scored by its edge-to-background ratio (strong-edge response ÷ flat-area response) on the filtered image. "
             f"{names[d_key]} scored highest ({snr[d_key]:.1f}) ahead of {names[runner]} ({snr[runner]:.1f}), so it gives the cleanest leaf/lesion outlines with the least noise response.")
    why_not = {
        "roberts": "2×2 kernel: very local, so it reacts strongly to noise and gives thin, broken edges.",
        "prewitt": "3×3 averaging kernel: smooth, but gives equal weight to all rows so edges are slightly less sharp than Sobel.",
        "sobel": "3×3 kernel with extra weight on the centre row/column: good smoothing and good edge localisation.",
        "laplacian_edge": "Second-derivative operator: very sensitive to noise and gives double edges.",
    }
    decisions.append({
        "stage": "Edge Detection", "input": "Filtered image (" + f_name + ")", "chosen": names[d_key], "output_key": d_key, "why": d_why,
        "metrics": [[names[k] + " edge/background", round(v, 1)] for k, v in snr.items()],
        "candidates": [_cand(names[k], k, k == d_key, ("Chosen: highest edge-to-background ratio. " if k == d_key else f"Not chosen: lower score ({snr[k]:.1f}). ") + why_not[k]) for k in ["roberts", "prewitt", "sobel", "laplacian_edge"]],
    })

    # ================= STEP 5 : THRESHOLDING (on the filtered gray image) =================
    _, global_t = cv2.threshold(gray_f, 120, 255, cv2.THRESH_BINARY)
    otsu_val, otsu = cv2.threshold(gray_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray_f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    eta = otsu_separability(gray_f, otsu_val)
    uneven = lighting_variation(gray_f, otsu_val)

    if uneven > 18:
        t_key, t_name = "adaptive", "Adaptive Thresholding"
        t_why = (f"Lighting is uneven: the brightness of the same kind of region (leaf or background) drifts by {uneven:.0f} levels across the image (above 18). "
                 f"One global level would wrongly cut shadowed or over-lit parts, so the threshold is computed locally for each 21×21 neighbourhood.")
    elif eta >= 0.5:
        t_key, t_name = "otsu", "Otsu Thresholding"
        t_why = (f"The histogram is clearly two-peaked (class separability {eta:.2f}, at least 0.5) and the lighting is even (variation {uneven:.0f}). "
                 f"Otsu automatically finds the best single level ({float(otsu_val):.0f}) between the two peaks, adapting to this photo instead of using a fixed number.")
    else:
        t_key, t_name = "adaptive", "Adaptive Thresholding"
        t_why = (f"The histogram is not clearly two-peaked (separability {eta:.2f}, below 0.5), so one global level - fixed or Otsu - would not separate "
                 f"the regions reliably. Adaptive thresholding decides pixel by pixel from the local neighbourhood.")
    mask = {"global": global_t, "otsu": otsu, "adaptive": adaptive}[t_key]
    decisions.append({
        "stage": "Thresholding", "input": "Filtered image (" + f_name + ") in grayscale", "chosen": t_name, "output_key": t_key, "why": t_why,
        "metrics": [["Otsu level", round(float(otsu_val), 1)], ["Histogram separability", round(eta, 2)], ["Lighting variation", round(uneven, 1)]],
        "candidates": [
            _cand("Global Thresholding (T = 120)", "global", False, "Not chosen: a fixed T = 120 ignores this image's own histogram; Otsu finds the best global level automatically."),
            _cand("Otsu Thresholding", "otsu", t_key == "otsu", "Chosen: two clear peaks and even light." if t_key == "otsu" else "Not chosen: " + ("lighting is uneven." if uneven > 18 else "histogram is not clearly two-peaked.")),
            _cand("Adaptive Thresholding", "adaptive", t_key == "adaptive", "Chosen: handles uneven light / weak peaks locally." if t_key == "adaptive" else "Not chosen: lighting is even and the histogram is two-peaked, and the local method would add speckle in flat leaf areas."),
        ]})

    # ================= STEP 6 : MORPHOLOGY (on the chosen binary mask) =================
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(mask, k, iterations=1)
    dilation = cv2.dilate(mask, k, iterations=1)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    morph_clean = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, k)

    tiny = max(4, int(0.0003 * total_px))
    specks = _small_components(mask, tiny)                       # tiny white blobs = noise specks
    inv = cv2.bitwise_not(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    holes = int(sum(1 for i in range(1, n) if i not in border and stats[i, cv2.CC_STAT_AREA] < tiny))  # tiny black holes
    LIM = 20
    if specks > LIM and holes <= LIM:
        m_key, m_name = "opening", "Opening (erosion → dilation)"
        m_why = (f"The mask has {specks} tiny isolated white specks but only {holes} small holes. Opening removes the specks "
                 f"and restores the size of the genuine leaf regions.")
    elif holes > LIM and specks <= LIM:
        m_key, m_name = "closing", "Closing (dilation → erosion)"
        m_why = (f"The mask has {holes} small black holes inside the objects but only {specks} stray specks. Closing fills the holes "
                 f"so the leaf region becomes solid.")
    else:
        m_key, m_name = "morph_clean", "Opening → Closing"
        m_why = (f"The mask has {specks} stray specks and {holes} small holes. " +
                 ("Both defects are present, so opening removes the specks and closing then fills the holes."
                  if specks > LIM and holes > LIM else
                  "Both counts are low, so only a light opening + closing is applied to smooth the boundaries."))
    decisions.append({
        "stage": "Morphology", "input": "Binary mask from " + t_name, "chosen": m_name, "output_key": m_key, "why": m_why,
        "metrics": [["Noise specks", specks], ["Small holes", holes], ["Structuring element", "3×3 ellipse"]],
        "candidates": [
            _cand("Erosion", "erosion", False, "Not chosen: it shrinks every region (including real lesions) and is only a building block of opening."),
            _cand("Dilation", "dilation", False, "Not chosen: it grows every region and would merge neighbouring spots; it is only a building block of closing."),
            _cand("Opening", "opening", m_key == "opening", "Chosen: removes specks." if m_key == "opening" else "Not chosen alone: " + ("holes also need filling." if m_key == "morph_clean" else "holes, not specks, are the main defect.")),
            _cand("Closing", "closing", m_key == "closing", "Chosen: fills holes." if m_key == "closing" else "Not chosen alone: " + ("specks also need removing." if m_key == "morph_clean" else "specks, not holes, are the main defect.")),
        ] + ([_cand("Opening → Closing", "morph_clean", True, "Chosen: both defects handled in one pass.")] if m_key == "morph_clean" else []),
    })

    images = {
        "original": img, "gray": gray, "hsv": hsv,
        "negative": negative, "contrast": contrast, "gamma": gamma,
        "gaussian": gaussian, "median": median, "laplacian": lap, "highboost": highboost,
        "roberts": roberts, "prewitt": prewitt, "sobel": sobel, "laplacian_edge": laplacian_edge,
        "global": global_t, "otsu": otsu, "adaptive": adaptive,
        "erosion": erosion, "dilation": dilation, "opening": opening, "closing": closing,
        "morph_clean": morph_clean,
    }
    numbers = {
        "min": int(gray.min()), "max": int(gray.max()),
        "mean": round(float(gray.mean()), 2), "otsu": round(float(otsu_val), 2),
        "gamma": round(gamma_val, 2), "decisions": decisions,
    }
    return images, numbers
