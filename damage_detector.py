"""Package damage detection - core model code (extracted from Package_Damage_Detection.ipynb).

Usage:
    import damage_detector as dd
    dd.load_models()                 # trains + saves to models/ the first time (~1-2 min)
    report = dd.inspect_package(bgr_image)
"""
import os, time
import numpy as np
import cv2
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

SEED = 42
IMG = 128
CLASSES = ['intact', 'crushed', 'torn', 'stained']
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
rf, cnn = None, None


def _background(rng):
    bg = np.full((IMG, IMG, 3), rng.uniform(70, 150), np.float32) + rng.uniform(-8, 8, 3)
    bg += rng.normal(0, 5, (IMG, IMG, 1))
    if rng.random() < 0.5:                                    # conveyor-belt slats
        bg[::int(rng.integers(10, 18))] -= 18
    return bg

def _box_corners(rng):
    cx, cy = IMG / 2 + rng.uniform(-6, 6, 2)
    w, h = rng.uniform(32, 46, 2)
    a = np.deg2rad(rng.uniform(-15, 15))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) @ R.T + [cx, cy]
    return pts + rng.uniform(-3, 3, pts.shape)                # slight perspective distortion

def _inside_point(rng, c, margin=0.2):
    u, v = rng.uniform(margin, 1 - margin, 2)
    top, bot = c[0] + u * (c[1] - c[0]), c[3] + u * (c[2] - c[3])
    return top + v * (bot - top)

def _pt(p):
    return tuple(int(round(v)) for v in p)

def _col(c):
    return tuple(float(v) for v in c)

def make_package(label, rng):
    img = _background(rng)
    corners = _box_corners(rng)
    centre = corners.mean(0)
    poly = list(corners)

    if label == 'crushed':                                    # push one corner inwards
        i = int(rng.integers(4))
        c, a, b = corners[i], corners[i - 1], corners[(i + 1) % 4]
        e1 = c + rng.uniform(0.12, 0.33) * (a - c)
        e2 = c + rng.uniform(0.12, 0.33) * (b - c)
        dent = c + rng.uniform(0.3, 0.55) * (centre - c)
        poly = poly[:i] + [e1, dent, e2] + poly[i + 1:]

    mask = np.zeros((IMG, IMG), np.uint8)
    cv2.fillPoly(mask, [np.int32(np.round(poly))], 255)

    colour = np.array([75, 135, 185], np.float32) + rng.uniform(-18, 18, 3)   # cardboard (BGR)
    box = np.ones((IMG, IMG, 3), np.float32) * colour
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    box *= (1 + 0.03 * np.sin(xx * rng.uniform(0.8, 1.4)))[..., None]       # corrugation
    box *= (1 + rng.normal(0, 0.04, (IMG, IMG)))[..., None]                 # paper grain
    gx, gy = rng.uniform(-0.0015, 0.0015, 2)
    box *= (1 + gx * (xx - IMG / 2) + gy * (yy - IMG / 2))[..., None]       # lighting gradient

    if rng.random() < 0.8:                                    # packing tape
        tape = np.zeros_like(mask)
        cv2.line(tape, _pt((corners[0] + corners[1]) / 2), _pt((corners[3] + corners[2]) / 2),
                 255, int(rng.integers(6, 11)))
        box[tape > 0] = box[tape > 0] * 1.12 + 12
    if rng.random() < 0.6:                                    # shipping label with barcode
        p = _inside_point(rng, corners, 0.3)
        lw, lh = int(rng.integers(10, 16)), int(rng.integers(7, 11))
        x0, y0 = int(p[0] - lw / 2), int(p[1] - lh / 2)
        cv2.rectangle(box, (x0, y0), (x0 + lw, y0 + lh), (235, 235, 235), -1)
        for k in range(x0 + 2, x0 + lw - 1, 2):
            if rng.random() < 0.7:
                cv2.line(box, (k, y0 + 2), (k, y0 + lh - 3), (30, 30, 30), 1)
    if rng.random() < 0.4:                                    # printed "this side up" arrows
        x, y = _pt(_inside_point(rng, corners, 0.25))
        for dx in (0, 5):
            cv2.arrowedLine(box, (x + dx, y + 6), (x + dx, y - 6), (40, 50, 60), 1, tipLength=0.4)

    if label == 'crushed':                                    # creases + shadow at the dent
        for _ in range(int(rng.integers(2, 4))):
            ang, L = rng.uniform(0, 2 * np.pi), rng.uniform(8, 18)
            cv2.line(box, _pt(dent), _pt(dent + L * np.array([np.cos(ang), np.sin(ang)])), _col(colour * 0.55), 1)
        shade = np.zeros((IMG, IMG), np.float32)
        cv2.circle(shade, _pt(dent), int(rng.integers(8, 14)), 1.0, -1)
        box *= (1 - 0.25 * cv2.GaussianBlur(shade, (0, 0), 4))[..., None]

    if label == 'torn':                                       # irregular holes with frayed edge
        for _ in range(int(rng.integers(1, 3))):
            c = _inside_point(rng, corners, 0.2)
            n, r = int(rng.integers(8, 15)), rng.uniform(3.5, 12)
            angs = np.sort(rng.uniform(0, 2 * np.pi, n))
            hole = np.int32(c + np.c_[np.cos(angs), np.sin(angs)] * (r * rng.uniform(0.5, 1.3, n))[:, None])
            cv2.fillPoly(box, [hole], _col(colour * rng.uniform(0.2, 0.35)))
            cv2.polylines(box, [hole], True, _col(np.minimum(colour * 1.35, 255)), 1)

    if label == 'stained':                                    # water stain with tide-line ring
        st = np.zeros((IMG, IMG), np.float32)
        for _ in range(int(rng.integers(1, 3))):
            ax = (int(rng.integers(8, 20)), int(rng.integers(8, 20)))
            cv2.ellipse(st, _pt(_inside_point(rng, corners, 0.25)), ax, rng.uniform(0, 180), 0, 360, 1.0, -1)
        st = np.clip(cv2.GaussianBlur(st, (0, 0), 3), 0, 1)
        ring = 4 * st * (1 - st)
        strength = rng.uniform(0.15, 0.32)                        # faint to strong stains
        box *= (1 - strength * st - 0.4 * strength * ring)[..., None]
        box[..., 0] *= (1 - 0.1 * st)

    m = mask > 0
    img[m] = box[m]
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    img += rng.normal(0, 3, img.shape)
    img *= rng.uniform(0.85, 1.15)                            # exposure
    return np.clip(img, 0, 255).astype(np.uint8)


KERNEL5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def segment_package(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = cv2.GaussianBlur(hsv[..., 1], (5, 5), 0)
    _, th = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, KERNEL5)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, KERNEL5, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:                                     # fallback: whole image is the package
        h, w = img.shape[:2]
        cnt = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], np.int32)
    else:
        cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask, cnt, th

def _lbp(gray):
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros(c.shape, np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        nb = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= (nb >= c).astype(np.uint8) << bit
    return np.pad(code, 1)

def _uniform_bin(code):                              # rotation-invariant uniform LBP -> 10 bins
    b = [(code >> i) & 1 for i in range(8)]
    transitions = sum(b[i] != b[(i + 1) % 8] for i in range(8))
    return sum(b) if transitions <= 2 else 9

LBP_LUT = np.array([_uniform_bin(i) for i in range(256)], np.uint8)

def lbp_image(gray):
    return LBP_LUT[_lbp(gray)]

def relative_luminance(img, inner):
    # LAB lightness divided by a least-squares plane fitted over the package -> removes uneven lighting
    L = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0], (7, 7), 0).astype(np.float32)
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    A = np.c_[xx[inner], yy[inner], np.ones(inner.sum())]
    keep = np.ones(len(A), bool)
    for _ in range(2):                                # refit ignoring outliers (stains, holes, labels)
        coef = np.linalg.lstsq(A[keep], L[inner][keep], rcond=None)[0]
        ratio = L[inner] / (A @ coef + 1e-6)
        keep = (ratio > 0.92) & (ratio < 1.12)
    return L / (coef[0] * xx + coef[1] * yy + coef[2] + 1e-6)


FEATURE_NAMES = (['area_ratio', 'solidity', 'extent', 'n_vertices', 'circularity', 'max_defect', 'sum_defect',
                  'dark_ratio', 'dark_blob', 'stain_ratio', 'edge_density', 'laplacian_var',
                  'H_mean', 'S_mean', 'V_mean', 'H_std', 'S_std', 'V_std'] + [f'lbp_{i}' for i in range(10)])

def convexity_defects(cnt):
    try:
        approx = cv2.approxPolyDP(cnt, 1.0, True)
        d = cv2.convexityDefects(approx, cv2.convexHull(approx, returnPoints=False))
        return approx, (d.reshape(-1, 4) if d is not None else np.zeros((0, 4), int))
    except cv2.error:
        return cnt, np.zeros((0, 4), int)

def extract_features(img):
    mask, cnt, _ = segment_package(img)
    inner = cv2.erode(mask, KERNEL5) > 0
    if inner.sum() < 50: inner = mask > 0
    area = cv2.contourArea(cnt) + 1e-6
    peri = cv2.arcLength(cnt, True) + 1e-6
    solidity = area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
    _, (rw, rh), _ = cv2.minAreaRect(cnt)
    extent = area / (rw * rh + 1e-6)
    n_vertices = len(cv2.approxPolyDP(cnt, 0.02 * peri, True))
    circularity = 4 * np.pi * area / peri ** 2
    _, d = convexity_defects(cnt)
    depths = d[:, 3] / 256.0 if len(d) else np.array([0.0])
    max_defect = depths.max() / np.sqrt(area)
    sum_defect = depths[depths > 2].sum() / np.sqrt(area)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    V = hsv[..., 2]
    medV = np.median(V[inner]) + 1e-6
    dark = ((V < 0.5 * medV) & inner).astype(np.uint8)
    dark_ratio = dark[inner].mean()
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    dark_blob = stats[1:, cv2.CC_STAT_AREA].max() / area if n > 1 else 0.0

    R = relative_luminance(img, inner)[inner]
    stain_ratio = np.mean((R < 0.88) & (R > 0.55))

    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    edge_density = (cv2.Canny(clahe, 60, 150)[inner] > 0).mean()
    lap_var = cv2.Laplacian(gray, cv2.CV_64F)[inner].var()
    hsv_stats = [hsv[..., i][inner].mean() for i in range(3)] + [hsv[..., i][inner].std() for i in range(3)]
    lbp_hist = np.bincount(lbp_image(gray)[inner], minlength=10) / inner.sum()

    return np.array([area / mask.size, solidity, extent, n_vertices, circularity, max_defect, sum_defect,
                     dark_ratio, dark_blob, stain_ratio, edge_density, lap_var] + hsv_stats + list(lbp_hist),
                    np.float32)


class PackageCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))

def to_tensor(imgs):
    imgs = np.stack([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]).astype(np.float32) / 255.0
    return torch.from_numpy(imgs).permute(0, 3, 1, 2)


def localize_damage(img):
    mask, cnt, _ = segment_package(img)
    inner = cv2.erode(mask, KERNEL5)
    area = cv2.contourArea(cnt) + 1e-6
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    regions, dmg = [], np.zeros(mask.shape, np.uint8)

    V = hsv[..., 2]; medV = np.median(V[inner > 0])
    dark = (((V < 0.5 * medV) & (inner > 0)) * 255).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    for c in cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) >= 12:
            regions.append(('tear', cv2.boundingRect(c))); cv2.drawContours(dmg, [c], -1, 255, -1)

    crush_zone = np.zeros(mask.shape, np.uint8)               # shadow around a dent is not a stain
    approx, d = convexity_defects(cnt)
    for s, e, f, depth in d:
        if depth / 256.0 > 0.05 * np.sqrt(area):
            tri = approx[[s, e, f], 0]
            regions.append(('crush', cv2.boundingRect(tri))); cv2.fillPoly(dmg, [tri], 255)
            cv2.circle(crush_zone, tuple(int(v) for v in approx[f, 0]), int(0.3 * np.sqrt(area)), 255, -1)

    R = relative_luminance(img, inner > 0)
    stain = (((R < 0.88) & (R >= 0.55) & (inner > 0)) * 255).astype(np.uint8)
    stain[(cv2.dilate(dark, KERNEL5, iterations=2) > 0) | (crush_zone > 0)] = 0
    stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    for c in cv2.findContours(stain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c) >= 100:
            regions.append(('stain', cv2.boundingRect(c))); cv2.drawContours(dmg, [c], -1, 255, -1)

    severity_pct = 100 * (dmg > 0).sum() / area
    return regions, cnt, severity_pct

BOX_COLOURS = {'tear': (0, 0, 255), 'stain': (255, 120, 0), 'crush': (0, 165, 255)}   # BGR

def draw_regions(img, regions, cnt, scale=3, damaged=True):
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.drawContours(out, [cnt * scale], -1, (0, 0, 255) if damaged else (0, 200, 0), 2)
    if damaged:
        for kind, (x, yb, w, h) in regions:
            col = BOX_COLOURS[kind]
            cv2.rectangle(out, (x * scale - 3, yb * scale - 3), ((x + w) * scale + 3, (yb + h) * scale + 3), col, 2)
            cv2.putText(out, kind, (x * scale, max(yb * scale - 7, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out


REGION_CLASS = {'tear': 'torn', 'stain': 'stained', 'crush': 'crushed'}


def analyze_package(img):
    """Raw model outputs for one image: RF, CNN and ensemble probabilities + classical damage localisation.
    Regions are in the 128 x 128 working frame (kind names 'tear' / 'stain' / 'crush')."""
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    p_rf = rf.predict_proba(extract_features(img)[None])[0]
    p_cnn = predict_cnn(to_tensor([img]))[0]
    regions, cnt, area_pct = localize_damage(img)
    mask, _, _ = segment_package(img)
    return {'image': img, 'p_rf': p_rf, 'p_cnn': p_cnn, 'p': 0.5 * p_rf + 0.5 * p_cnn,
            'regions': regions, 'contour': cnt, 'mask': mask, 'damaged_area_pct': float(area_pct)}


def inspect_package(img):
    """Simple argmax inspection (kept for the notebook; the app uses inspection.run_inspection)."""
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    p = 0.5 * rf.predict_proba(extract_features(img)[None])[0] + 0.5 * predict_cnn(to_tensor([img]))[0]
    k = int(p.argmax())
    damaged = CLASSES[k] != 'intact'
    regions, cnt, sev = localize_damage(img)
    if not damaged:
        severity = 'None'
    else:
        severity = 'Minor' if sev < 3 else 'Moderate' if sev < 8 else 'Severe'
    return {'decision': 'REJECT' if damaged else 'ACCEPT', 'damage_type': CLASSES[k], 'confidence': round(float(p[k]), 3),
            'severity': severity, 'damaged_area_pct': round(float(sev), 1) if damaged else 0.0,
            'regions': regions if damaged else [], 'probabilities': dict(zip(CLASSES, np.round(p, 3).tolist())),
            'annotated': draw_regions(img, regions, cnt, damaged=damaged)}

def predict_cnn(x):
    cnn.eval()
    with torch.no_grad():
        return torch.cat([F.softmax(cnn(x[i:i + 256]), 1) for i in range(0, len(x), 256)]).numpy()


def train_models(n_per_class=400, epochs=25, log=print):
    """Generate the synthetic dataset, train the Random Forest + CNN and save them to models/."""
    global rf, cnn
    np.random.seed(SEED); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X = np.array([make_package(c, rng) for c in CLASSES for _ in range(n_per_class)])
    y = np.repeat(np.arange(len(CLASSES)), n_per_class)
    log(f'Generated {len(X)} images, extracting features...')
    Fx = np.stack([extract_features(im) for im in X])
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Fx, y)
    log('Random Forest trained, training CNN...')

    cnn = PackageCNN()
    Xt, yt = to_tensor(X), torch.from_numpy(y).long()
    BS = 64
    opt = torch.optim.Adam(cnn.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * int(np.ceil(len(y) / BS)))
    for ep in range(epochs):
        cnn.train()
        perm = np.random.permutation(len(y))
        for b in range(0, len(perm), BS):
            xb, yb = Xt[perm[b:b + BS]], yt[perm[b:b + BS]]
            if np.random.rand() < 0.5: xb = xb.flip(3)
            if np.random.rand() < 0.5: xb = xb.flip(2)
            loss = F.cross_entropy(cnn(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        log(f'CNN epoch {ep + 1}/{epochs}')
    cnn.eval()

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest.joblib'))
    torch.save(cnn.state_dict(), os.path.join(MODEL_DIR, 'cnn.pt'))
    log(f'Models saved to {MODEL_DIR}')


def load_models(log=print):
    """Load saved models, or train them if they don't exist yet."""
    global rf, cnn
    rf_path, cnn_path = os.path.join(MODEL_DIR, 'random_forest.joblib'), os.path.join(MODEL_DIR, 'cnn.pt')
    if not (os.path.exists(rf_path) and os.path.exists(cnn_path)):
        train_models(log=log)
        return
    rf = joblib.load(rf_path)
    cnn = PackageCNN()
    cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
    cnn.eval()


if __name__ == '__main__':
    t0 = time.time()
    train_models()
    print(f'Done in {time.time() - t0:.0f} s')
