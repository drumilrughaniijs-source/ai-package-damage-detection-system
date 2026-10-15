"""Shared Streamlit UI: model loading, cached inspection, the result view, pipeline view and small helpers."""
import glob
import hashlib
import os

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import damage_detector as dd
import evaluation as ev
import inspection as ins
import real_inspector as ri
import storage

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(ROOT, 'samples')
CLASS_COLOURS = {'intact': '#16A34A', 'crushed': '#7C3AED', 'torn': '#DC2626', 'stained': '#2563EB'}
DECISION_COLOUR = {'ACCEPT': 'green', 'REJECT': 'red', 'NEEDS REVIEW': 'orange'}
DECISION_ICON = {'ACCEPT': ':material/check_circle:', 'REJECT': ':material/cancel:',
                 'NEEDS REVIEW': ':material/pending_actions:'}


# ---------------------------------------------------------------- models & state
@st.cache_resource(show_spinner='Loading AI models (the first run downloads the CLIP model, ~600 MB)...')
def _load_models():
    ins.load_models()
    return True


def load_engines():
    try:
        _load_models()
    except Exception as e:
        st.error(f'The AI models could not be loaded ({type(e).__name__}: {e}). Run setup.bat once with an internet '
                 'connection, then restart the app.', icon=':material/error:')
        st.stop()


def init_state():
    st.session_state.setdefault('threshold', ins.DEFAULT_THRESHOLD)
    st.session_state.setdefault('session', [])        # [(id, decision)] inspected in this browser session
    st.session_state.setdefault('reports', {})        # inspection key -> report (keeps IDs stable across reruns)
    st.session_state.setdefault('seed', 1)


def sidebar():
    """Settings (rendered on every run so the slider keeps its value); returns a slot for the session counters."""
    with st.sidebar:
        st.subheader('Decision settings', icon=':material/tune:')
        st.slider('Rejection threshold', 0.50, 0.85, step=0.05, key='threshold',
                  help='A package is accepted only if the model gives P(intact) of at least this value. '
                       'Higher = stricter quality control (more rejections).')
        t = st.session_state.threshold
        st.caption(f'ACCEPT if P(intact) ≥ {t + ins.REVIEW_MARGIN:.2f} · REJECT if ≤ {t - ins.REVIEW_MARGIN:.2f} · '
                   f'in between → NEEDS REVIEW. Calibrated default: {ins.DEFAULT_THRESHOLD:.2f}')
        st.subheader('AI engines', icon=':material/memory:')
        st.markdown(':green-badge[:material/check: Real photos] CLIP ViT-B/32 + U²-Net  \n'
                    ':green-badge[:material/check: Synthetic] Random Forest + CNN')
        stats = st.container()
        st.caption('Image & Video Processing (AIDLL 5052) mini project')
    return stats


def session_stats(slot):
    s = st.session_state.session
    with slot:
        st.subheader('This session', icon=':material/history:')
        with st.container(horizontal=True):
            st.metric('Inspected', len(s))
            st.metric('Accepted', sum(d == 'ACCEPT' for _, d in s))
        with st.container(horizontal=True):
            st.metric('Rejected', sum(d == 'REJECT' for _, d in s))
            st.metric('Review', sum(d == 'NEEDS REVIEW' for _, d in s))


# ---------------------------------------------------------------- inspection
def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sample_files():
    return sorted(glob.glob(os.path.join(SAMPLE_DIR, '*.jpg')))


def sample_label(path):
    """Short label without the class word, e.g. 'crushed_fedex_box.jpg' -> 'Fedex box'."""
    name = os.path.splitext(os.path.basename(path))[0]
    rest = name.split('_', 3)[-1] if name.startswith('not_a_package') else name.split('_', 1)[-1]
    return rest.replace('_', ' ').capitalize()


@st.cache_data(show_spinner=False)
def sample_thumb(path, size=220):
    """Square centre-cropped thumbnail (RGB) so the gallery grid lines up."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    m = min(h, w)
    img = img[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
    return rgb(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))


@st.cache_data(max_entries=12, show_spinner=False)
def _analyze(img, synthetic):
    return ins.analyze(img, synthetic)


def inspect(img, filename, source, synthetic=False, save=True):
    """Run (or reuse) an inspection. The same image + threshold + source gives the same report, so reruns do not
    create duplicate history records. Raises ins.InspectionError with a user-readable message."""
    key = (hashlib.sha1(img.tobytes()).hexdigest(), img.shape, st.session_state.threshold, filename, source, synthetic)
    reports = st.session_state.reports
    if key in reports:
        return reports[key]
    r = ins.decide(_analyze(img, synthetic), st.session_state.threshold, filename, source)
    if save:
        err = storage.save(r)
        if err:
            st.toast(err, icon=':material/warning:')
    st.session_state.session.append((r['id'], r['decision']))
    reports[key] = r
    while len(reports) > 12:                              # bound memory: forget the oldest cached reports
        reports.pop(next(iter(reports)))
    return r


def slim(r):
    """Report without the heavy in-memory model maps (for lists such as batch / conveyor results)."""
    return {k: v for k, v in r.items() if k != '_analysis'}


def safe_name(name):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in os.path.splitext(name)[0])[:60] or 'image'


# ---------------------------------------------------------------- result view
def prob_chart(probs, height=150):
    df = pd.DataFrame({'Class': list(probs), 'Probability': list(probs.values())})
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=20).encode(
        x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%', title=None)),
        y=alt.Y('Class:N', sort=list(probs), title=None),
        color=alt.Color('Class:N', scale=alt.Scale(domain=list(CLASS_COLOURS), range=list(CLASS_COLOURS.values())),
                        legend=None),
        tooltip=['Class', alt.Tooltip('Probability:Q', format='.1%')]).properties(height=height)


DECISION_HEX = {'ACCEPT': '#16A34A', 'REJECT': '#DC2626', 'NEEDS REVIEW': '#F59E0B'}


def count_chart(counts, colours, height=150):
    """Horizontal bar chart of counts with fixed category colours (decisions or damage types)."""
    df = pd.DataFrame({'Category': list(counts.index), 'Packages': list(counts.values)})
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X('Packages:Q', axis=alt.Axis(tickMinStep=1, title=None)),
        y=alt.Y('Category:N', sort=list(counts.index), title=None),
        color=alt.Color('Category:N', scale=alt.Scale(domain=list(colours), range=list(colours.values())), legend=None),
        tooltip=['Category', 'Packages']).properties(height=height)


def decision_banner(r):
    d = r['decision']
    if d == 'ACCEPT':
        st.success('**ACCEPT** · the package passed inspection', icon=DECISION_ICON[d])
    elif d == 'REJECT':
        sev = f" ({r['severity'].lower()})" if r['severity'] else ''
        st.error(f"**REJECT** · {r['damage_type']} damage{sev}", icon=DECISION_ICON[d])
    else:
        what = 'no package detected - retake the photo' if r['invalid_image'] else 'a person should check this package'
        st.warning(f'**NEEDS REVIEW** · {what}', icon=DECISION_ICON[d])


def decision_badge(r):
    return f":{DECISION_COLOUR[r['decision']]}-badge[{r['decision']}]"


def views(r):
    """[(name, RGB image, legend)] - only views backed by real model output are offered."""
    a = r['_analysis']
    if a['synthetic']:
        s = ins.SYNTH_SCALE
        orig = cv2.resize(a['image'], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        seg = np.where(cv2.resize(a['mask'], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)[..., None] > 0,
                       orig, (orig * 0.3).astype(np.uint8))
        out = [('Annotated', rgb(r['annotated']),
                'Outline colour = decision. Boxes = damage regions found by the classical localiser '
                '(dark blobs = torn, luminance drop = stained, convexity defect = crushed).'),
               ('Original', rgb(orig), 'Generated package (128 px, shown at 3x).'),
               ('Package mask', rgb(seg),
                'Package region from Otsu threshold on saturation + morphology. This is the package, not the damage.')]
        return out
    out = [('Annotated', rgb(r['annotated']),
            'Outline colour = decision (green accept, red reject, orange review). Boxes = suspected damage regions '
            '(violet crushed, red torn, blue stained), drawn only where the model localised damage. '
            'Circles = dent candidates from convexity defects.'),
           ('Original', rgb(a['image']), 'Photo as analysed (resized to max 768 px).'),
           ('Package mask', rgb(ri.segmentation_overlay(a)),
            'Bright area = package region found by U²-Net salient-object segmentation. This shows the package, '
            'not the damage.' + (' No distinct object was found, so the whole frame is used.'
                                 if a['seg_fallback'] else ''))]
    if not r['invalid_image']:
        out.append(('Damage map', rgb(ri.damage_overlay(a)),
                    'Colour = CLIP damage probability of the windows (25% of the image) covering each point, shown '
                    'only where it exceeds 0.5. Window-level, so it marks areas, not exact pixels.'))
    return out


def image_panel(r, key):
    vs = views(r)
    names = ['Compare'] + [v[0] for v in vs]
    choice = st.pills('View', names, default='Compare', key=f'{key}_view', label_visibility='collapsed') or 'Compare'
    if choice == 'Compare':
        c1, c2 = st.columns(2)
        c1.image(vs[1][1], caption='Original', width='stretch')
        c2.image(vs[0][1], caption='Annotated result', width='stretch')
        st.caption(vs[0][2])
    else:
        name, im, legend = next(v for v in vs if v[0] == choice)
        st.image(im, width='stretch')
        st.caption(legend)


def prediction_card(r):
    with st.container(border=True):
        st.markdown('**Model prediction** :gray[· what the AI outputs]')
        with st.container(horizontal=True):
            st.metric('P(damaged)', f"{r['p_damaged']:.0%}", border=True,
                      help='Probability of any damage = 1 - P(intact), from the model.')
            st.metric('Most likely damage', (r['damage_type'] or '-').capitalize(), border=True,
                      help='Shown when P(intact) is below the threshold. Share = this type\'s part of the damage '
                           'probability.')
        st.altair_chart(prob_chart(r['probabilities']), width='stretch')
        ag = r['agreement']
        if r['engine'] == 'synthetic':
            same = ag['random_forest'] == ag['cnn']
            st.caption(f"{':green[:material/check:]' if same else ':orange[:material/warning:]'} Random Forest: "
                       f"**{ag['random_forest']}** · CNN: **{ag['cnn']}**")
        else:
            st.caption(f"P(intact): whole image **{ag['whole_image_p_intact']:.0%}** · close-up windows "
                       f"**{ag['local_windows_p_intact']:.0%}** · package score **{r['package_score']:.0%}**")


def decision_card(r):
    with st.container(border=True):
        st.markdown("**Inspection decision** :gray[· business rule applied to the prediction]")
        with st.container(horizontal=True):
            st.metric('Severity', r['severity'] or ('-' if r['decision'] == 'ACCEPT' else 'Not assessed'), border=True,
                      help='Extent of the localised damage, independent of model confidence.')
            st.metric('Damaged area', '-' if r['damaged_area_pct'] is None else f"{r['damaged_area_pct']:.0f}%",
                      border=True, help=f"Measured as: {r['area_method']}. Only shown when damage was localised.")
        with st.container(horizontal=True):
            st.metric('Threshold', f"{r['threshold']:.2f}", border=True, help='Accept if P(intact) ≥ threshold.')
            st.metric('Processing time', f"{r['processing_ms'] / 1000:.2f} s", border=True,
                      help='Measured model + decision time for this image on this computer.')
        st.markdown('\n'.join(f'- {line}' for line in r['explanation']))
        if r['localisation_note']:
            st.caption(f":material/info: {r['localisation_note']}")
        for n in r['notes']:
            st.caption(f':material/info: {n}')
        if r['quality']:
            q = r['quality']
            st.caption(f"Image quality: sharpness {q['sharpness']:.0f} · brightness {q['brightness']:.0f} · "
                       f"{q['width']}x{q['height']} px")


def downloads(r, key):
    with st.container(horizontal=True):
        st.download_button('Annotated image', ins.annotated_png(r), f"{r['id']}_annotated.png", 'image/png',
                           icon=':material/download:', key=f'{key}_png')
        st.download_button('JSON report', ins.report_json(r), f"{r['id']}_report.json", 'application/json',
                           icon=':material/data_object:', key=f'{key}_json')


def show_report(r, key='rep'):
    decision_banner(r)
    st.caption(f"Inspection **{r['id']}** · {r['timestamp'].replace('T', ' ')} · {r['filename']} · {r['source']} · "
               f"{r['model']}")
    left, right = st.columns([1.35, 1], gap='large')
    with left:
        image_panel(r, key)
        if r['regions']:
            st.dataframe(pd.DataFrame([{'Region': i + 1, 'Type': g['type'],
                                        'Damage prob.': g['score'], 'x': g['box'][0], 'y': g['box'][1],
                                        'Width': g['box'][2], 'Height': g['box'][3]} for i, g in enumerate(r['regions'])]),
                         hide_index=True, width='stretch', column_config={
                             'Damage prob.': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent')})
        downloads(r, key)
    with right:
        prediction_card(r)
        decision_card(r)


# ---------------------------------------------------------------- image-processing pipeline views
def _heat_rgb(bgr, heat):
    norm = np.clip(heat / (heat.max() + 1e-6), 0, 1)
    return rgb(cv2.addWeighted(bgr, 0.45, cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), 0.55, 0))


def real_pipeline(r):
    a = r['_analysis']
    bgr, mask = a['image'], a['mask']
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(clahe, (5, 5), 0), 60, 150)
    contour = rgb(bgr).copy()
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if cnts and not a['seg_fallback']:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(contour, [cv2.convexHull(c)], -1, (255, 0, 255), 2)
        cv2.drawContours(contour, [c], -1, (0, 255, 0), 2)
        for x, y in ri.find_dents(mask):
            cv2.circle(contour, (x, y), 8, (255, 140, 0), 3)
    return [('1. Photo (max 768 px)', rgb(bgr)),
            ('2. U²-Net saliency map', (a['saliency'] * 255).astype(np.uint8)),
            ('3. Threshold + morphology → package mask', mask),
            ('4. Contour, convex hull, dent candidates', contour),
            ('5. CLIP package score per window', _heat_rgb(bgr, np.clip(a['package_map'], 0, 1))),
            ('6. CLIP damage map (> 0.5 coloured)', rgb(ri.damage_overlay(a))),
            ('7. CLAHE contrast enhancement', clahe),
            ('8. Canny edges on the package', edges * (mask > 0))]


def synthetic_pipeline(r):
    img = r['_analysis']['image']
    mask, cnt, th = dd.segment_package(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    contour = rgb(img).copy()
    cv2.drawContours(contour, [cv2.convexHull(cnt)], -1, (255, 0, 255), 1)
    cv2.drawContours(contour, [cnt], -1, (0, 255, 0), 1)
    inner = cv2.erode(mask, dd.KERNEL5) > 0
    lum = np.clip((dd.relative_luminance(img, inner) - 0.5) / 0.7 * 255, 0, 255).astype(np.uint8) * inner
    steps = [('1. Original', rgb(img)), ('2. HSV saturation', hsv[..., 1]), ('3. Otsu + morphology', th),
             ('4. Contour & convex hull', contour), ('5. CLAHE enhanced', clahe),
             ('6. Canny edges', cv2.Canny(clahe, 60, 150)), ('7. LBP texture', dd.lbp_image(gray) * 28),
             ('8. Illumination-corrected luminance', lum)]
    return [(t, cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)) for t, im in steps]


def show_pipeline(r, key='pipe'):
    with st.expander('Image-processing pipeline: every intermediate step', icon=':material/account_tree:'):
        steps = synthetic_pipeline(r) if r['_analysis']['synthetic'] else real_pipeline(r)
        for row in (steps[:4], steps[4:]):
            for col, (title, im) in zip(st.columns(4), row):
                col.image(im, caption=title, width='stretch')


def result_row(r):
    """Flat dict for tables / CSV exports."""
    return {'ID': r['id'], 'Time': r['timestamp'].replace('T', ' '), 'File': r['filename'], 'Source': r['source'],
            'Decision': r['decision'], 'Damage type': r['damage_type'] or '-', 'Severity': r['severity'] or '-',
            'P(intact)': r['p_intact'], 'Damaged area %': r['damaged_area_pct'], 'Threshold': r['threshold'],
            'Processing s': round(r['processing_ms'] / 1000, 2),
            'Review reasons': ' '.join(r['review_reasons'])}


def truth_of(path):
    return ev.sample_truth(path)


TRUTH_BADGE = {'intact': 'green', 'crushed': 'violet', 'torn': 'red', 'stained': 'blue', 'not_package': 'gray'}


def truth_badge(path):
    t = truth_of(path)
    return f":{TRUTH_BADGE[t]}-badge[{t.replace('_', ' ')}]"
