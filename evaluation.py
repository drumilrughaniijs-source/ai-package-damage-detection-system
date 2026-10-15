"""Model evaluation on labelled data. Every number shown in the app's evaluation page is computed here, live.

Data sources
* Synthetic held-out set: freshly generated with EVAL_SEED. The deployed RF + CNN were trained by
  damage_detector.train_models() on images generated with SEED = 42, so these images were never seen in training.
* Real photos: eval_data/<class>/*.jpg (classes: intact, crushed, torn, stained, not_package) if that folder exists -
  this is where an independent, held-out real test set belongs. Otherwise the bundled samples/ are used, which is NOT
  an independent test: the real-photo thresholds were calibrated on those photos.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import damage_detector as dd
import inspection as ins

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(ROOT, 'eval_data')
SAMPLE_DIR = os.path.join(ROOT, 'samples')
EVAL_SEED = 2026
TRUTH_CLASSES = ins.CLASSES + ['not_package']


def synthetic_set(n_per_class, seed=EVAL_SEED):
    """[(truth, image)] generated with a seed different from the training seed."""
    if seed == dd.SEED:
        raise ValueError('The evaluation seed must differ from the training seed.')
    rng = np.random.default_rng(seed)
    return [(c, dd.make_package(c, rng)) for c in dd.CLASSES for _ in range(n_per_class)]


def sample_truth(path):
    name = os.path.basename(path).lower()
    if name.startswith('not_a_package'):
        return 'not_package'
    if name.startswith('dented'):
        return 'crushed'
    return name.split('_')[0]


def real_dataset():
    """(list of (truth, path), description, independent: bool)."""
    files = []
    if os.path.isdir(EVAL_DIR):
        for c in TRUTH_CLASSES:
            for ext in ins.ALLOWED_EXTENSIONS:
                files += [(c, f) for f in glob.glob(os.path.join(EVAL_DIR, c, f'*.{ext}'))]
    if files:
        return sorted(files), f'{len(files)} labelled photos from eval_data/', True
    files = [(sample_truth(f), f) for f in sorted(glob.glob(os.path.join(SAMPLE_DIR, '*.jpg')))]
    return files, f'{len(files)} bundled sample photos (samples/)', False


def summarize(rows):
    """rows: dicts with truth, decision, predicted (intact / damage type / NEEDS REVIEW), invalid, model_label."""
    df = pd.DataFrame(rows)
    pkg = df[df.truth != 'not_package']
    decided = pkg[pkg.decision != 'NEEDS REVIEW']
    out = {'n': len(df), 'n_packages': len(pkg), 'n_decided': len(decided),
           'review_rate': float((pkg.decision == 'NEEDS REVIEW').mean()) if len(pkg) else None}

    damaged, intact = pkg[pkg.truth != 'intact'], pkg[pkg.truth == 'intact']
    out['far'] = float((damaged.decision == 'ACCEPT').mean()) if len(damaged) else None    # damaged parcel accepted
    out['frr'] = float((intact.decision == 'REJECT').mean()) if len(intact) else None      # good parcel rejected
    out['accuracy'] = float((decided.predicted == decided.truth).mean()) if len(decided) else None
    out['decision_accuracy'] = (float(((decided.decision == 'ACCEPT') == (decided.truth == 'intact')).mean())
                                if len(decided) else None)

    labels = [c for c in ins.CLASSES if c in set(decided.truth) | set(decided.predicted)]
    if len(decided):
        prec, rec, f1, sup = precision_recall_fscore_support(decided.truth, decided.predicted, labels=labels,
                                                             zero_division=0)
        out['per_class'] = pd.DataFrame({'Class': labels, 'Precision': prec, 'Recall': rec, 'F1-score': f1,
                                         'Support': sup})
        out['macro_f1'] = float(np.mean(f1[sup > 0])) if (sup > 0).any() else None
    cm_labels = [c for c in TRUTH_CLASSES if c in set(df.truth)]
    pred_labels = ins.CLASSES + ['NEEDS REVIEW']
    cm = confusion_matrix(df.truth, df.predicted, labels=cm_labels + [p for p in pred_labels if p not in cm_labels])
    out['confusion'] = pd.DataFrame(cm, index=cm_labels + [p for p in pred_labels if p not in cm_labels],
                                    columns=cm_labels + [p for p in pred_labels if p not in cm_labels]
                                    ).loc[cm_labels, pred_labels]
    if (df.truth == 'not_package').any():
        neg = df[df.truth == 'not_package']
        out['invalid_detection_rate'] = float(neg.invalid.mean())
    if 'model_label' in df:
        out['model_accuracy'] = float((pkg.model_label == pkg.truth).mean()) if len(pkg) else None
    return out
