"""One inspection, end to end - the single source of truth used by every page (single, batch, conveyor, evaluation).

    image bytes --load_image()--> BGR image --analyze()--> raw model outputs --decide()--> report

* analyze() runs the expensive models (and is cached by the UI).
* decide() is cheap and deterministic: it turns model outputs + the configured rejection threshold into
  ACCEPT / REJECT / NEEDS REVIEW, with the reasons, severity, localisation and the annotated image.

Nothing here invents numbers: every probability, area and region comes from a model output, and when something was
not measured (e.g. damage could not be localised) the report says so instead of filling in a value.
"""
import io
import json
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import damage_detector as dd
import real_inspector as ri

CLASSES = ri.CLASSES
DECISIONS = ['ACCEPT', 'REJECT', 'NEEDS REVIEW']

# ---- decision settings (see README "Decision logic")
DEFAULT_THRESHOLD = 0.65   # ACCEPT when P(intact) >= threshold (calibrated default for the real-photo engine)
REVIEW_MARGIN = 0.05       # P(intact) this close to the threshold -> NEEDS REVIEW instead of a coin-flip decision
PACKAGE_MIN = 0.70         # package score below this -> package presence uncertain -> NEEDS REVIEW
PACKAGE_INVALID = 0.35     # package score below this -> "no package detected" (invalid image)
CONFLICT_GAP = 0.30        # whole-image vs local-window P(intact) differing this much across the threshold -> review
MIN_SHARPNESS = 40.0       # variance of Laplacian at 512 px; real photos measured 136-7500, blurred copies <= 10
MIN_BRIGHTNESS = 40.0      # mean grey level; real photos 86-185, under-exposed copies 13-25
MAX_BRIGHTNESS = 235.0
MIN_SIDE = 160             # px; smaller photos are too coarse for the window analysis

# ---- severity = extent of the localised damage (kept separate from model confidence)
REAL_SEVERITY = [(40, 'Minor'), (75, 'Moderate'), (101, 'Severe')]     # % of package windows flagged as damaged
SYNTH_SEVERITY = [(3, 'Minor'), (8, 'Moderate'), (101, 'Severe')]      # % of package pixels in damage regions

# ---- upload validation
MAX_UPLOAD_MB = 15
MAX_PIXELS = 40_000_000
ALLOWED_FORMATS = {'JPEG', 'MPO', 'PNG', 'WEBP', 'BMP'}
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'bmp']

SYNTH_MODEL_INFO = 'Random Forest (28 classical IVP features) + 4-layer CNN ensemble, trained on synthetic packages'
SYNTH_SCALE = 3            # synthetic results are drawn at 3x the 128 px working size


class InspectionError(Exception):
    """An error with a message that can be shown to the user as is."""


# ---------------------------------------------------------------- input
def load_image(data, name='image'):
    """Validate an uploaded file and decode it to a BGR image. Raises InspectionError with a readable message."""
    if not data:
        raise InspectionError(f'"{name}" is empty.')
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise InspectionError(f'"{name}" is larger than {MAX_UPLOAD_MB} MB.')
    try:
        with Image.open(io.BytesIO(data)) as im:          # reads the header only - cheap, no full decode
            fmt, (w, h) = im.format, im.size
    except Exception:
        raise InspectionError(f'"{name}" is not a readable image (unsupported or corrupted file).') from None
    if fmt not in ALLOWED_FORMATS:
        raise InspectionError(f'"{name}" is a {fmt} file. Supported formats: JPG, PNG, WEBP, BMP.')
    if w * h > MAX_PIXELS:
        raise InspectionError(f'"{name}" is {w}x{h} px - images above {MAX_PIXELS // 1_000_000} megapixels are refused.')
    if min(w, h) < 32:
        raise InspectionError(f'"{name}" is only {w}x{h} px - too small to inspect.')
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)   # also applies EXIF rotation
    if img is None:
        raise InspectionError(f'"{name}" could not be decoded - the file may be corrupted.')
    return img


# ---------------------------------------------------------------- models
def load_models():
    dd.load_models(log=lambda *_: None)
    ri.load_model()


def analyze(img, synthetic=False):
    """Run the models (expensive). Returns the raw outputs plus the measured analysis time."""
    t0 = time.perf_counter()
    try:
        a = dd.analyze_package(img) if synthetic else ri.analyze(img)
    except Exception as e:
        engine = 'synthetic-package' if synthetic else 'real-photo'
        raise InspectionError(f'The {engine} model failed on this image ({type(e).__name__}: {e}).') from e
    a['synthetic'] = synthetic
    a['analysis_ms'] = (time.perf_counter() - t0) * 1000
    return a


# ---------------------------------------------------------------- decision
def _probs(p):
    """Class -> probability rounded to 3 decimals (as Python floats, so JSON shows 0.275, not 0.2750000059)."""
    return {c: round(float(v), 3) for c, v in zip(CLASSES, p)}


def _band(value, bands):
    return next(name for limit, name in bands if value < limit)


def _final(p_intact, threshold, review):
    if review:
        return 'NEEDS REVIEW'
    return 'ACCEPT' if p_intact >= threshold else 'REJECT'


def _quality_issues(q):
    issues = []
    if q['sharpness'] < MIN_SHARPNESS:
        issues.append(f"Image is blurry (sharpness {q['sharpness']:.0f}, minimum {MIN_SHARPNESS:.0f}).")
    if q['brightness'] < MIN_BRIGHTNESS:
        issues.append(f"Image is too dark (brightness {q['brightness']:.0f}, minimum {MIN_BRIGHTNESS:.0f}).")
    if q['brightness'] > MAX_BRIGHTNESS:
        issues.append(f"Image is over-exposed (brightness {q['brightness']:.0f}, maximum {MAX_BRIGHTNESS:.0f}).")
    if min(q['width'], q['height']) < MIN_SIDE:
        issues.append(f"Resolution too low ({q['width']}x{q['height']} px, minimum {MIN_SIDE} px on the short side).")
    return issues


def _decide_real(a, threshold):
    p = a['p']
    p_int, g, l, pkg = float(p[0]), float(a['p_global'][0]), float(a['p_local'][0]), a['package_score']
    review, notes = [], []
    invalid = pkg < PACKAGE_INVALID
    if invalid:
        review.append(f'No package detected: package score {pkg:.0%} (needs {PACKAGE_MIN:.0%}). Retake the photo '
                      'with the parcel clearly visible.')
    elif pkg < PACKAGE_MIN:
        review.append(f'Package presence is uncertain: package score {pkg:.0%} (needs {PACKAGE_MIN:.0%}).')
    review += _quality_issues(a['quality'])
    if not invalid:
        if (g >= threshold) != (l >= threshold) and abs(g - l) > CONFLICT_GAP:
            review.append(f'Whole-image and close-up analyses disagree: P(intact) {g:.0%} vs {l:.0%}.')
        if abs(p_int - threshold) < REVIEW_MARGIN:
            review.append(f'P(intact) = {p_int:.0%} is within {REVIEW_MARGIN:.0%} of the {threshold:.0%} threshold.')
    decision = _final(p_int, threshold, review)
    leaning_damaged = p_int < threshold and not invalid
    k = 1 + int(p[1:].argmax())

    regions, dents, area, severity, loc_note = [], [], None, None, None
    if leaning_damaged:
        regions = ri.damage_regions(a, p)
        if CLASSES[k] == 'crushed' and not a['seg_fallback']:
            dents = ri.find_dents(a['mask'])
        area = a['damage_coverage_pct']
        if area is None or area == 0:
            area = None
            loc_note = ('Damage was detected from the image as a whole, but no individual window was scored as damaged, '
                        'so the damage could not be localised and severity was not assessed.')
        else:
            severity = _band(area, REAL_SEVERITY)
            if not regions:
                loc_note = 'Damaged windows are scattered; no connected damage region was large enough to outline.'
        if a['seg_fallback']:
            notes.append('Segmentation found no distinct object, so the whole frame was treated as the package.')
    elif not invalid:
        loc_note = 'No damage localisation: the package is not predicted to be damaged.'
    else:
        loc_note = 'No localisation: no package was detected.'

    return {'engine': 'real-photo', 'model': ri.MODEL_INFO, 'p': p, 'decision': decision, 'review_reasons': review,
            'invalid_image': invalid, 'package_score': round(pkg, 3), 'damage_index': k if leaning_damaged else None,
            'agreement': {'whole_image_p_intact': round(g, 3), 'local_windows_p_intact': round(l, 3)},
            'quality': {k2: round(v, 1) if isinstance(v, float) else v for k2, v in a['quality'].items()},
            'severity': severity, 'damaged_area_pct': None if area is None else round(area, 1),
            'area_method': 'share of package windows with CLIP P(damaged) > 0.5', 'regions': regions,
            'region_frame': [int(a['image'].shape[1]), int(a['image'].shape[0])], 'dents': [list(d) for d in dents],
            'localisation_note': loc_note, 'notes': notes,
            'annotated': ri.annotate(a, decision, regions, dents)}


def _annotate_synthetic(a, decision, regions):
    s = SYNTH_SCALE
    out = cv2.resize(a['image'], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    cv2.drawContours(out, [a['contour'] * s], -1, ri.DECISION_COLOURS[decision], 2)
    for r in regions:
        x, y, w, h = r['box']
        col = ri.TYPE_COLOURS[r['type']]
        cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), col, 2)
        cv2.putText(out, r['type'], (x, max(y - 7, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), ri.DECISION_COLOURS[decision], 4)
    return out


def _decide_synthetic(a, threshold):
    p = a['p']
    p_int = float(p[0])
    rf_k, cnn_k = int(a['p_rf'].argmax()), int(a['p_cnn'].argmax())
    review = []
    if rf_k != cnn_k:
        review.append(f'The two models disagree: Random Forest says {CLASSES[rf_k]}, CNN says {CLASSES[cnn_k]}.')
    if abs(p_int - threshold) < REVIEW_MARGIN:
        review.append(f'P(intact) = {p_int:.0%} is within {REVIEW_MARGIN:.0%} of the {threshold:.0%} threshold.')
    decision = _final(p_int, threshold, review)
    leaning_damaged = p_int < threshold
    k = 1 + int(p[1:].argmax())

    regions, area, severity, loc_note = [], None, None, None
    if leaning_damaged:
        s = SYNTH_SCALE
        regions = [{'type': dd.REGION_CLASS[kind], 'score': None, 'box': [int(x * s), int(y * s), int(w * s), int(h * s)]}
                   for kind, (x, y, w, h) in a['regions']]
        if regions:
            area = a['damaged_area_pct']
            severity = _band(area, SYNTH_SEVERITY)
        else:
            loc_note = ('The classifiers predict damage, but the rule-based localiser found no damage region, '
                        'so severity was not assessed.')
    else:
        loc_note = 'No damage localisation: the package is not predicted to be damaged.'

    return {'engine': 'synthetic', 'model': SYNTH_MODEL_INFO, 'p': p, 'decision': decision, 'review_reasons': review,
            'invalid_image': False, 'package_score': None, 'damage_index': k if leaning_damaged else None,
            'agreement': {'random_forest': CLASSES[rf_k], 'cnn': CLASSES[cnn_k],
                          'random_forest_p': _probs(a['p_rf']),
                          'cnn_p': _probs(a['p_cnn'])},
            'quality': None, 'severity': severity, 'damaged_area_pct': None if area is None else round(area, 1),
            'area_method': 'share of package pixels inside classical damage regions', 'regions': regions,
            'region_frame': [dd.IMG * SYNTH_SCALE, dd.IMG * SYNTH_SCALE], 'dents': [],
            'localisation_note': loc_note, 'notes': [], 'annotated': _annotate_synthetic(a, decision, regions)}


def _explain(r, threshold):
    """Plain-language reasons for the final decision."""
    p_int, d = r['p_intact'], r['decision']
    lines = []
    if r['invalid_image']:
        lines.append('The photo does not appear to show a package, so no accept/reject decision was made.')
    else:
        side = 'meets' if p_int >= threshold else 'is below'
        lines.append(f'The model gives P(intact) = {p_int:.0%}, which {side} the rejection threshold of {threshold:.0%}.')
        if r['damage_type']:
            lines.append(f"Most likely damage type: {r['damage_type']} ({r['type_share']:.0%} of the damage probability).")
    if d == 'NEEDS REVIEW':
        lines.append('Sent to manual review because: ' + ' '.join(r['review_reasons']))
    elif d == 'ACCEPT':
        lines.append('No review condition was triggered, so the package is accepted.')
    else:
        lines.append('No review condition was triggered, so the package is rejected.')
    return lines


def decide(a, threshold=DEFAULT_THRESHOLD, filename='image', source='Upload'):
    """Turn model outputs into a complete inspection report (cheap; re-run whenever the threshold changes)."""
    t0 = time.perf_counter()
    r = _decide_synthetic(a, threshold) if a['synthetic'] else _decide_real(a, threshold)
    p = r.pop('p')
    k = r.pop('damage_index')
    r.update({
        'id': uuid.uuid4().hex[:10].upper(), 'timestamp': datetime.now().isoformat(timespec='seconds'),
        'filename': filename, 'source': source, 'threshold': round(float(threshold), 2),
        'review_margin': REVIEW_MARGIN,
        'probabilities': _probs(p),
        'p_intact': round(float(p[0]), 3), 'p_damaged': round(float(1 - p[0]), 3),
        'top_class': CLASSES[int(p.argmax())], 'top_class_probability': round(float(p.max()), 3),
        'damage_type': CLASSES[k] if k is not None else None,
        'type_share': round(float(p[k] / max(p[1:].sum(), 1e-6)), 3) if k is not None else None,
        'analysis_ms': round(a['analysis_ms'], 1),
    })
    r['explanation'] = _explain(r, threshold)
    r['processing_ms'] = round(a['analysis_ms'] + (time.perf_counter() - t0) * 1000, 1)
    r['_analysis'] = a                                     # in-memory only (views); never exported
    return r


def run_inspection(img, threshold=DEFAULT_THRESHOLD, filename='image', source='Upload', synthetic=False):
    return decide(analyze(img, synthetic), threshold, filename, source)


def predicted_label(r):
    """Label used for evaluation: intact / damage type / NEEDS REVIEW."""
    return {'ACCEPT': 'intact', 'REJECT': r['damage_type']}.get(r['decision'], 'NEEDS REVIEW')


# ---------------------------------------------------------------- export
EXPORT_KEYS = ['id', 'timestamp', 'filename', 'source', 'decision', 'review_reasons', 'explanation', 'invalid_image',
               'damage_type', 'type_share', 'severity', 'damaged_area_pct', 'area_method', 'p_intact', 'p_damaged',
               'probabilities', 'top_class', 'top_class_probability', 'threshold', 'review_margin', 'package_score',
               'agreement', 'quality', 'regions', 'region_frame', 'dents', 'localisation_note', 'notes',
               'analysis_ms', 'processing_ms', 'engine', 'model']


def export_dict(r):
    out = {k: r.get(k) for k in EXPORT_KEYS}
    out['annotated_image'] = f"{r['id']}_annotated.png"
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def report_json(r):
    return json.dumps(export_dict(r), indent=2)


def annotated_png(r):
    ok, buf = cv2.imencode('.png', r['annotated'])
    if not ok:
        raise InspectionError('Could not encode the annotated image.')
    return buf.tobytes()
