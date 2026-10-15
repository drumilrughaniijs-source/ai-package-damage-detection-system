"""Real-photo package analysis: deep segmentation + vision-language model (CLIP, zero-shot) + classical IVP.

The RF + CNN in damage_detector.py were trained on synthetic packages only, so they do not transfer to real
camera photos. For real photos this module combines:

  1. U2-Net-p salient-object segmentation (OpenCV DNN)  -> package mask and outline
  2. OpenCLIP ViT-B/32 (trained on 2 billion real image-text pairs, LAION-2B), zero-shot:
     the photo and text descriptions ("a crushed cardboard box", "a water damaged parcel", ...) are mapped into
     the same embedding space and compared - no task-specific training images needed.
       * is it a package at all?           (package prompts vs. 14 "other object" prompts)
       * intact / crushed / torn / stained  (prompt ensembles, whole image + mirrored copy + 16 local windows)
  3. 49 small windows scored by CLIP, weighted by package coverage -> coarse damage-probability map
  4. Convexity defects of the package contour                       -> dent candidates on crushed packages

This module only produces model outputs. The ACCEPT / REJECT / NEEDS REVIEW decision is made in inspection.py.

Usage:
    import real_inspector as ri
    ri.load_model()
    a = ri.analyze(bgr_image)
"""
import os
import numpy as np
import cv2
import torch
from PIL import Image

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')           # plain HTTP download is more reliable on Windows
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import open_clip

MODEL_NAME, PRETRAINED = 'ViT-B-32', 'laion2b_s34b_b79k'
MODEL_INFO = f'OpenCLIP {MODEL_NAME} ({PRETRAINED}) zero-shot + U2-Net-p segmentation'
SEG_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'u2netp.onnx')
CLASSES = ['intact', 'crushed', 'torn', 'stained']
LOCAL_WEIGHT = 0.5             # fusion: (1-w) * whole-image scores + w * package-weighted local window scores
LOGIT_SCALE = 100.0
WINDOW_DAMAGE_THRESHOLD = 0.5  # a window counts as "damaged" when CLIP gives P(damaged) > 0.5 for it
MAX_SIDE = 768

TEMPLATES = ['a photo of {}.', 'a close-up photo of {}.', 'a picture of {}.', '{}']
CLASS_PROMPTS = {
    'intact': ['an intact cardboard box', 'an undamaged shipping box in good condition',
               'a new, clean parcel with straight edges', 'a perfect sealed carton', 'a neat cardboard box'],
    'crushed': ['a crushed cardboard box', 'a dented, squashed shipping box', 'a parcel with a smashed, caved-in corner',
                'a deformed, collapsed cardboard box', 'a crumpled, battered parcel'],
    'torn': ['a torn cardboard box', 'a ripped shipping box with a hole', 'a parcel with torn, ripped open packaging',
             'a punctured cardboard box', 'shredded torn cardboard'],
    'stained': ['a water damaged cardboard box', 'a wet, stained shipping box', 'a parcel with dirty water stains',
                'a soggy, discoloured cardboard box', 'a cardboard box with water marks and mould'],
}
PACKAGE_PROMPTS = ['a cardboard box', 'a parcel package', 'a shipping box', 'a carton', 'a damaged cardboard box',
                   'cardboard packaging']
OTHER_PROMPTS = ['a person', 'a building', 'a street', 'a document with text', 'a telephone booth', 'an animal', 'a car',
                 'furniture', 'a room', 'a landscape', 'food', 'an electronic device', 'a sign', 'a poster']

model, preprocess, class_emb, gate_emb, seg_net = None, None, None, None, None


def _encode_text(tokenizer, phrases):
    with torch.no_grad():
        e = model.encode_text(tokenizer([t.format(p) for p in phrases for t in TEMPLATES]))
    e = e / e.norm(dim=-1, keepdim=True)
    e = e.mean(0)                                        # prompt ensembling: average over phrasings + templates
    return e / e.norm()


def load_model():
    """Download (first run, ~600 MB) and load CLIP + U2-Net, then pre-compute the text embeddings."""
    global model, preprocess, class_emb, gate_emb, seg_net
    if model is not None:
        return
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    m, _, prep = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    m.eval()
    model, preprocess = m, prep
    tok = open_clip.get_tokenizer(MODEL_NAME)
    class_emb = torch.stack([_encode_text(tok, CLASS_PROMPTS[c]) for c in CLASSES])
    gate_emb = torch.stack([_encode_text(tok, PACKAGE_PROMPTS)] + [_encode_text(tok, [o]) for o in OTHER_PROMPTS])
    seg_net = cv2.dnn.readNetFromONNX(SEG_MODEL)


# ---------------------------------------------------------------- segmentation (U2-Net-p salient object detection)
def saliency(bgr):
    """U2-Net-p saliency map (0..1) at the image's resolution."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (rgb / max(rgb.max(), 1) - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    seg_net.setInput(x.transpose(2, 0, 1)[None].astype(np.float32))
    s = seg_net.forward()[0, 0]
    s = (s - s.min()) / (s.max() - s.min() + 1e-6)
    return cv2.resize(s, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def segment_package(bgr):
    """Package mask (uint8 0/255) = largest salient blob. Returns (mask, saliency, fallback) where fallback=True
    means nothing salient was found and the whole frame is used (typical for close-ups)."""
    sal = saliency(bgr)
    m = (sal > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + st[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
    fallback = m.mean() < 0.06
    if fallback:
        m[:] = 1
    return m * 255, sal, bool(fallback)


# ---------------------------------------------------------------- CLIP scoring
def embed(rgb_images, batch=32):
    """L2-normalised CLIP image embeddings for a list of RGB uint8 arrays."""
    out = []
    with torch.no_grad():
        for i in range(0, len(rgb_images), batch):
            x = torch.stack([preprocess(Image.fromarray(im)) for im in rgb_images[i:i + batch]])
            e = model.encode_image(x)
            out.append(e / e.norm(dim=-1, keepdim=True))
    return torch.cat(out)


def _scores(emb):
    p_cls = (LOGIT_SCALE * emb @ class_emb.T).softmax(-1)
    p_pkg = (LOGIT_SCALE * emb @ gate_emb.T).softmax(-1)[:, 0]
    return p_cls.numpy(), p_pkg.numpy()


def _crops(rgb, win=0.45, steps=4):
    """Overlapping windows (win x image size) on a steps x steps grid."""
    h, w = rgb.shape[:2]
    ch, cw = int(h * win), int(w * win)
    boxes = [(int(x), int(y), cw, ch) for y in np.linspace(0, h - ch, steps) for x in np.linspace(0, w - cw, steps)]
    return boxes, [rgb[y:y + ch, x:x + cw] for x, y, cw, ch in boxes]


def _accumulate(shape, boxes, values, weights):
    """Spread per-window scores back onto the pixels each window covers (weighted average)."""
    h, w = shape
    acc, cnt = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for (x, y, cw, ch), v, wt in zip(boxes, values, weights):
        acc[y:y + ch, x:x + cw] += v * wt
        cnt[y:y + ch, x:x + cw] += wt
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1e-6), 0).astype(np.float32)
    return cv2.GaussianBlur(out, (0, 0), max(h, w) / 40)


def find_dents(mask):
    """Classical cue for crushing: deep convexity defects of the package outline -> list of (x, y) points."""
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not cnts or mask.min() > 0:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.004 * cv2.arcLength(cnt, True), True)
    try:
        d = cv2.convexityDefects(approx, cv2.convexHull(approx, returnPoints=False))
    except cv2.error:
        return []
    if d is None:
        return []
    size = np.sqrt(cv2.contourArea(cnt))
    return [tuple(int(v) for v in approx[f, 0]) for s, e, f, depth in d.reshape(-1, 4) if depth / 256.0 > 0.045 * size]


def image_quality(bgr):
    """Sharpness (variance of the Laplacian) and mean brightness, measured at a fixed 512 px scale."""
    s = 512 / max(bgr.shape[:2])
    g = cv2.cvtColor(cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return {'sharpness': float(cv2.Laplacian(g, cv2.CV_64F).var()), 'brightness': float(g.mean()),
            'width': int(bgr.shape[1]), 'height': int(bgr.shape[0])}


# ---------------------------------------------------------------- main analysis
def analyze(bgr):
    """All model outputs for one real photo (no decision is made here)."""
    quality = image_quality(bgr)
    s = MAX_SIDE / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    shape = rgb.shape[:2]

    mask, sal, seg_fallback = segment_package(bgr)
    boxes_c, crops_c = _crops(rgb, 0.45, 4)                 # 16 medium windows -> classification (fusion)
    boxes_l, crops_l = _crops(rgb, 0.25, 7)                 # 49 small windows  -> localisation
    emb = embed([rgb, rgb[:, ::-1].copy()] + crops_c + crops_l)   # + mirrored copy (test-time augmentation)
    p_cls, p_pkg = _scores(emb)
    nc = 2 + len(boxes_c)

    p_global = p_cls[:2].mean(0)
    w = np.clip(p_pkg[2:nc] * 1.5, 0, 1)[:, None]           # medium windows weighted by "is this a box?"
    p_local = (p_cls[2:nc] * w).sum(0) / (w.sum() + 1e-6)
    p = (1 - LOCAL_WEIGHT) * p_global + LOCAL_WEIGHT * p_local

    # localisation: P(damaged) of each small window, weighted by how much of it lies on the package
    pc_l = p_cls[nc:]
    win_damage = 1 - pc_l[:, 0]
    cover = np.array([(mask[y:y + ch, x:x + cw] > 0).mean() for x, y, cw, ch in boxes_l])
    damage_map = _accumulate(shape, boxes_l, win_damage, np.clip(cover, 0.02, 1) ** 2) * (mask > 0)
    on_pkg = cover > 0.5
    coverage = float((win_damage[on_pkg] > WINDOW_DAMAGE_THRESHOLD).mean() * 100) if on_pkg.any() else None

    return {'image': bgr, 'quality': quality, 'mask': mask, 'saliency': sal, 'seg_fallback': seg_fallback,
            'p': p, 'p_global': p_global, 'p_local': p_local, 'package_score': float(p_pkg[:2].mean()),
            'window_boxes': boxes_l, 'window_damage': win_damage, 'window_cover': cover,
            'window_probs': pc_l, 'damage_map': damage_map, 'damage_coverage_pct': coverage,
            'package_map': _accumulate(shape, boxes_l, p_pkg[nc:], np.ones(len(boxes_l)))}


def damage_regions(a, global_p, max_regions=3):
    """Suspected damage regions = connected areas on the package where the window damage map exceeds 0.5.
    Each region is typed by the windows inside it (blended 50/50 with the global damage-type probabilities)."""
    hot = ((a['damage_map'] > WINDOW_DAMAGE_THRESHOLD) & (a['mask'] > 0)).astype(np.uint8)
    cnts = sorted(cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)
    centres = [(x + cw / 2, y + ch / 2) for x, y, cw, ch in a['window_boxes']]
    regions = []
    for c in cnts[:max_regions]:
        if cv2.contourArea(c) < 0.01 * hot.size:
            continue
        inside = np.array([cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0 for cx, cy in centres])
        local = a['window_probs'][inside].mean(0) if inside.any() else global_p
        mix = 0.5 * local[1:] / local[1:].sum() + 0.5 * global_p[1:] / global_p[1:].sum()
        x, y, w, h = cv2.boundingRect(c)
        regions.append({'type': CLASSES[1 + int(mix.argmax())], 'score': round(float(1 - local[0]), 3),
                        'box': [int(x), int(y), int(w), int(h)]})
    return regions


# ---------------------------------------------------------------- drawing
TYPE_COLOURS = {'crushed': (237, 58, 124), 'torn': (38, 38, 220), 'stained': (235, 99, 37)}   # BGR: violet, red, blue
DECISION_COLOURS = {'ACCEPT': (74, 163, 22), 'REJECT': (38, 38, 220), 'NEEDS REVIEW': (12, 140, 234)}   # BGR


def _label(out, text, x, y, col, fs, t):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t - 1))
    ty = y - 8 if y - th - 14 > 0 else y + th + 10
    cv2.rectangle(out, (x, ty - th - 6), (x + tw + 10, ty + 6), col, -1)
    cv2.putText(out, text, (x + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)


def annotate(a, decision, regions, dents):
    """Result image: package outline coloured by decision, suspected damage regions, dent candidates."""
    out = a['image'].copy()
    t = max(2, out.shape[1] // 320)
    fs = max(0.5, out.shape[1] / 1300)
    col = DECISION_COLOURS[decision]
    if not a['seg_fallback']:
        cv2.drawContours(out, cv2.findContours(a['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, col, t)
    for x, y in dents:
        cv2.circle(out, (x, y), 4 * t, (255, 255, 255), t + 2)
        cv2.circle(out, (x, y), 4 * t, TYPE_COLOURS['crushed'], t)
    for r in regions:
        rc = TYPE_COLOURS[r['type']]
        x, y, w, h = r['box']
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), t + 2)
        cv2.rectangle(out, (x, y), (x + w, y + h), rc, t)
        _label(out, f"suspected {r['type']} {r['score'] * 100:.0f}%", x, y, rc, fs, t)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), col, 3 * t)
    return out


def colourbar(width, lo=WINDOW_DAMAGE_THRESHOLD, hi=1.0, height=34):
    """Legend strip for the damage map (JET colours from lo to hi)."""
    bar = np.full((height, width, 3), 24, np.uint8)
    x0, x1 = 60, width - 60
    ramp = cv2.applyColorMap(np.linspace(0, 255, x1 - x0).astype(np.uint8)[None].repeat(10, 0), cv2.COLORMAP_JET)
    bar[4:14, x0:x1] = ramp
    for v, x in ((lo, x0), ((lo + hi) / 2, (x0 + x1) // 2), (hi, x1)):
        cv2.putText(bar, f'{v:.2f}', (x - 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(bar, '<0.5', (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    return bar


def damage_overlay(a):
    """Damage-probability map on an absolute scale: only areas with window P(damaged) > 0.5 are coloured."""
    bgr, dm = a['image'], a['damage_map']
    norm = np.clip((dm - WINDOW_DAMAGE_THRESHOLD) / (1 - WINDOW_DAMAGE_THRESHOLD), 0, 1)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.where(dm > WINDOW_DAMAGE_THRESHOLD, 0.55, 0.0)[..., None]
    out = (bgr * (1 - alpha) + cmap * alpha).astype(np.uint8)
    if not a['seg_fallback']:
        cv2.drawContours(out, cv2.findContours(a['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1,
                         (255, 255, 255), max(1, out.shape[1] // 400))
    return np.vstack([out, colourbar(out.shape[1])])


def segmentation_overlay(a):
    """Package region (U2-Net) bright, background dimmed, outline in white."""
    bgr, mask = a['image'], a['mask']
    out = np.where(mask[..., None] > 0, bgr, (bgr * 0.3).astype(np.uint8))
    cv2.drawContours(out, cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 255), 2)
    return out
