import pandas as pd
import streamlit as st

import inspection as ins

st.header('Goal', icon=':material/flag:')
st.markdown('Inspect parcels from an ordinary photo and decide **ACCEPT**, **REJECT** or **NEEDS REVIEW** - with the '
            'damage type (crushed, torn, water-stained), where the damage is, how severe it is, and the reason for the '
            'decision. Uncertain cases go to a person instead of being guessed.')

st.header('Two engines, one decision layer', icon=':material/hub:')
left, right = st.columns(2, gap='large')
with left.container(border=True, height='stretch'):
    st.subheader(':material/photo_camera: Real photos')
    st.markdown('''
- **U²-Net** salient-object network (OpenCV DNN) separates the package from the background
- **CLIP ViT-B/32** vision-language model (pre-trained on 2 billion image-text pairs, LAION-2B) scores the photo
  **zero-shot** against text prompts ("a crushed cardboard box", "a wet, stained parcel", ...)
- **Package gate**: package prompts vs. 14 other-object prompts ("a person", "a sign", ...)
- **Global + local fusion**: whole image + mirrored copy + 16 overlapping windows
- **Damage map**: 49 small windows scored by CLIP, restricted to the package mask
- **Convexity defects** of the package outline mark dent candidates
- **Image quality**: sharpness (variance of Laplacian), brightness, resolution
''')
with right.container(border=True, height='stretch'):
    st.subheader(':material/auto_awesome: Synthetic packages')
    st.markdown('''
- Procedural generator draws boxes with tape, labels, crushing, tears and water stains
- **Classical IVP**: HSV/LAB conversion, Otsu threshold, morphology, contours, convex hull, CLAHE, Canny,
  LBP texture, illumination correction
- **28 hand-crafted features → Random Forest**, plus a **4-layer CNN** on raw pixels; probabilities averaged
- **Rule-based localisation**: dark blobs (tears), luminance drop (stains), convexity defects (crushing)
- Used by the synthetic generator, the conveyor simulation and the held-out evaluation
''')

st.header('Decision logic', icon=':material/rule:')
st.markdown('The models only produce probabilities. One shared decision layer (`inspection.py`) turns them into the '
            'final business decision, identically for single, batch and conveyor inspections:')
st.table(pd.DataFrame({
    'Check': ['No package', 'Package uncertain', 'Image quality', 'Models disagree', 'Too close to call',
              'Accept / reject'],
    'Rule': [f'Package score < {ins.PACKAGE_INVALID:.0%} → NEEDS REVIEW ("no package detected")',
             f'Package score < {ins.PACKAGE_MIN:.0%} → NEEDS REVIEW',
             f'Sharpness < {ins.MIN_SHARPNESS:.0f}, brightness outside {ins.MIN_BRIGHTNESS:.0f}-{ins.MAX_BRIGHTNESS:.0f}, '
             f'short side < {ins.MIN_SIDE} px → NEEDS REVIEW',
             f'Real photos: whole-image and close-up P(intact) on opposite sides of the threshold and > '
             f'{ins.CONFLICT_GAP:.0%} apart. Synthetic: Random Forest and CNN predict different classes → NEEDS REVIEW',
             f'|P(intact) - threshold| < {ins.REVIEW_MARGIN:.2f} → NEEDS REVIEW',
             'P(intact) ≥ threshold → ACCEPT, otherwise REJECT with the most likely damage type'],
}).set_index('Check'))
st.markdown('''
**Kept separate on purpose**
- **Confidence** = the model's probabilities (P(intact), P(damaged), class probabilities).
- **Threshold** = your quality-control policy (sidebar), not a model property.
- **Severity** = the *extent* of the localised damage: share of package windows the model scores as damaged
  (real photos: < 40 % minor, < 75 % moderate, otherwise severe) or share of package pixels inside damage regions
  (synthetic: < 3 % / < 8 %). If the damage could not be localised, severity is reported as *not assessed*.
''')

st.header('Real-photo pipeline', icon=':material/account_tree:')
st.mermaid_chart('''
flowchart TD
    A[Photo] --> V[Validate file<br/>format, size, decode]
    V --> Q[Image quality<br/>sharpness, brightness]
    V --> C[U2-Net saliency<br/>package mask]
    V --> E[CLIP image encoder<br/>full + flip + 16 + 49 windows]
    T[Text prompts] --> F[CLIP text encoder]
    E --> G[Probabilities<br/>package score, P intact, damage types]
    F --> G
    Q --> D{Decision layer}
    G --> D
    D -- review condition --> R[NEEDS REVIEW + reasons]
    D -- P intact >= threshold --> ACC[ACCEPT]
    D -- P intact < threshold --> REJ[REJECT + damage type]
    C --> L[Damage map inside mask<br/>+ convexity defects]
    G --> L
    L --> S[Regions, damaged area, severity]
''', width='stretch')

st.header('Results', icon=':material/analytics:')
st.markdown('Accuracy, precision, recall, F1, confusion matrices and false acceptance / rejection rates are computed '
            'live on the **Model evaluation** page, on a synthetic set the models never saw and on labelled real '
            'photos. No accuracy figure in this app is typed in by hand.')

st.header('Why two engines?', icon=':material/upgrade:')
st.markdown('''
The first version learned only from computer-generated boxes, so a real photo (different lighting, background,
printing, camera) looked nothing like its training data - a **domain shift** you can measure on the Model evaluation
page. Labelled photos of real damaged parcels were not available in quantity, so real photos are handled by a
**foundation model** (CLIP) that has seen billions of real photos and is told the classes in words
(**zero-shot learning**). Classical image processing is still used for segmentation clean-up, contours, convexity
defects, enhancement and image-quality checks.
''')

st.header('Limitations', icon=':material/report:')
st.markdown('''
- The real-photo thresholds were calibrated on a small set of web photos; no large labelled real dataset was available,
  so real-photo accuracy has not been measured on an independent set (add one under `eval_data/`).
- CLIP is not trained for damage inspection specifically: subtle damage on a distant box, unusual packaging (bags,
  envelopes) and damage hidden from the camera are hard. These often end up in NEEDS REVIEW.
- The damage map is window-based (each window covers 25 % of the image), so it marks areas, not exact pixels.
- Torn packages: no real torn-box photos were available for testing.
- The conveyor is a software simulation; no hardware is connected.
''')

st.header('Tech stack', icon=':material/build:')
st.markdown(':blue-badge[Python] :blue-badge[OpenCV] :blue-badge[PyTorch] :blue-badge[OpenCLIP] '
            ':blue-badge[U²-Net (ONNX)] :blue-badge[scikit-learn] :blue-badge[SQLite] :blue-badge[Streamlit]')
