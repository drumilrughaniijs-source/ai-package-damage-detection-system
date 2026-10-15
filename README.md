# AI Package Damage Inspector

Image & Video Processing (AIDLL 5052) mini project. A photo of a parcel goes in; the system decides
**ACCEPT**, **REJECT** or **NEEDS REVIEW**, names the damage (crushed, torn, water-stained), shows where it is,
estimates how severe it is and explains why.

## Install and run (Windows, Python 3.14)

```
setup.bat        first time only: pip install -r requirements.txt + downloads the CLIP model (~600 MB)
run_app.bat      starts the app at http://localhost:8501
```

or manually:

```
py -3.14 -m pip install -r requirements.txt
py -3.14 -m streamlit run app.py
```

## Pages

| Page | What it does |
|---|---|
| Inspect | Sample photos, upload, camera or synthetic generator → decision, model prediction, explanation, image views (compare / annotated / original / package mask / damage map), JSON + annotated image download, processing pipeline |
| Batch inspection | Up to 40 photos at once; failed files listed with reasons; counts per decision; CSV, ZIP of annotated images, JSON |
| Conveyor simulation | Software simulation: package enters → AI inspection → decision → routed to accept / reject / manual-review lane |
| Inspection history | Persistent (SQLite `data/inspections.db` + `data/images/`): filters, search, details with stored images, re-download, CSV export, clear |
| Model evaluation | Live accuracy, precision, recall, F1, confusion matrix, false acceptance / rejection rate, review rate |
| How it works | Architecture, decision rules (read from the code constants), limitations |

## AI pipeline

```
image bytes ─ load_image()  validate format/size/pixels, decode (EXIF rotation applied)
            ─ analyze()     model outputs only (cached)
                 real photo:  U²-Net mask · CLIP package score · CLIP class probabilities (whole image + mirror
                              + 16 windows) · 49-window damage map · convexity-defect dents · sharpness/brightness
                 synthetic:   Random Forest (28 IVP features) + CNN probabilities · rule-based damage regions
            ─ decide()      threshold + review rules → decision, reasons, severity, regions, annotated image
```

All pages call the same `inspection.py`, so single, batch, conveyor and evaluation results are consistent.

### Decision logic (`inspection.py`)

| Condition | Result |
|---|---|
| Package score < 0.35 | NEEDS REVIEW - no package detected |
| Package score < 0.70 | NEEDS REVIEW - package presence uncertain |
| Sharpness < 40, brightness < 40 or > 235, short side < 160 px | NEEDS REVIEW - image quality |
| Whole-image vs close-up P(intact) on opposite sides of the threshold and > 0.30 apart (real) / RF and CNN disagree (synthetic) | NEEDS REVIEW - models disagree |
| \|P(intact) − threshold\| < 0.05 | NEEDS REVIEW - too close to call |
| otherwise P(intact) ≥ threshold | ACCEPT, else REJECT with the most likely damage type |

* **Confidence** = model probabilities. **Threshold** = policy (sidebar, default 0.65). **Severity** = extent of the
  *localised* damage (share of package windows scored damaged, or share of package pixels in damage regions).
  If damage could not be localised, severity is "not assessed" - no value is invented.
* Package segmentation (where the package is) and the damage map (where the model thinks damage is) are separate
  views, each with a legend. Damage regions are only drawn where the model actually scored windows above 0.5.

## Evaluation

* **Synthetic held-out set**: generated with seed 2026; the deployed models were trained on seed 42 images.
* **Real photos**: put labelled photos in `eval_data/intact/`, `eval_data/crushed/`, `eval_data/torn/`,
  `eval_data/stained/`, `eval_data/not_package/` for an independent test. Without that folder, the 11 bundled
  samples are used and the page warns that they are **not independent** (the thresholds were calibrated on them).
* The notebook's 80/20 accuracy numbers describe the notebook's models; `damage_detector.train_models()` trains the
  deployed models on all generated images, so the app no longer quotes those numbers.

## Files

| File | What |
|---|---|
| `app.py`, `app_pages/`, `ui.py` | Streamlit app |
| `inspection.py` | Validation, decision rules, explanations, report export (single source of truth) |
| `real_inspector.py` | Real-photo engine (CLIP + U²-Net + localisation + quality) |
| `damage_detector.py` | Synthetic generator, classical IVP features, Random Forest + CNN |
| `storage.py` | Persistent inspection history (SQLite) |
| `evaluation.py` | Datasets and metrics for the evaluation page |
| `models/` | Trained RF / CNN + `u2netp.onnx` |
| `samples/` | Real demo photos (licences in `samples/ATTRIBUTION.md`) |
| `data/` | Created at runtime: inspection history database and images |

## Demo script (about 6 minutes)

1. **Inspect → Sample photos.** Pick *Fedex box* (REJECT, crushed): show *Compare*, *Package mask* vs *Damage map*
   (with legend), the *Model prediction* card vs the *Inspection decision* card, and the explanation.
2. Pick *Closed box* (ACCEPT), then *Phone booth* (NEEDS REVIEW - no package) and *Packed box* (NEEDS REVIEW -
   190x127 px photo, too low resolution).
3. Move the **Rejection threshold** in the sidebar to 0.85 and re-inspect *Closed box*: the decision changes; open
   the pipeline expander to walk through the IVP steps.
4. **Upload photo / Camera**: a real box from the room; squash a corner for REJECT, cover the lens or shake the
   camera to show the blur / darkness review path.
5. **Batch inspection → Inspect all sample photos**, export CSV / ZIP.
6. **Conveyor simulation → Run conveyor** (three lanes), then **Inspection history** (filter, select a row).
7. **Model evaluation**: run the synthetic held-out evaluation, then the real-photo evaluation with the
   domain-shift check (why real photos need the CLIP engine).

## Limitations

* No large labelled real damaged-parcel dataset was available: real-photo thresholds were calibrated on ~23 web
  photos and real-photo accuracy has not been measured on an independent set.
* CLIP is a general model, not a trained damage detector: subtle or hidden damage and unusual packaging are hard and
  often end up in NEEDS REVIEW. No real torn-box photos were available for testing.
* The damage map is window-based (windows are 25 % of the image): it marks areas, not exact pixels, and individual
  regions can be typed differently from the overall damage type.
* The conveyor is a simulation; no hardware is connected.
