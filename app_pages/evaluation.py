import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import damage_detector as dd
import evaluation as ev
import inspection as ins
import ui

st.markdown('All numbers on this page are computed live by running the full inspection pipeline (models + decision '
            f'rules at the current threshold, **{st.session_state.threshold:.2f}**) on labelled images. Nothing is '
            'typed in by hand. Evaluation runs are not saved to the inspection history.')


def run_items(items, synthetic, key):
    """items: [(truth, name, image)] -> rows for ev.summarize (+ shows progress)."""
    rows = []
    bar = st.progress(0.0, text='Evaluating...')
    for i, (truth, name, img) in enumerate(items):
        try:
            r = ins.decide(ui._analyze(img, synthetic), st.session_state.threshold, name, 'Evaluation')
            rows.append({'truth': truth, 'name': name, 'decision': r['decision'], 'predicted': ins.predicted_label(r),
                         'invalid': r['invalid_image'], 'p_intact': r['p_intact'], 'reasons': ' '.join(r['review_reasons'])})
        except ins.InspectionError as e:
            rows.append({'truth': truth, 'name': name, 'decision': 'ERROR', 'predicted': 'ERROR', 'invalid': False,
                         'p_intact': np.nan, 'reasons': str(e)})
        bar.progress((i + 1) / len(items), text=f'Evaluating {i + 1}/{len(items)}')
    bar.empty()
    return rows


def pct(v):
    return '-' if v is None else f'{v * 100:.1f}%'


def confusion_chart(cm):
    d = cm.reset_index().melt(id_vars='index', var_name='Predicted', value_name='Count').rename(columns={'index': 'Actual'})
    base = alt.Chart(d).encode(x=alt.X('Predicted:N', sort=list(cm.columns), axis=alt.Axis(labelAngle=0)),
                               y=alt.Y('Actual:N', sort=list(cm.index)))
    rect = base.mark_rect(cornerRadius=3).encode(color=alt.Color('Count:Q', scale=alt.Scale(scheme='oranges'),
                                                                 legend=None), tooltip=['Actual', 'Predicted', 'Count'])
    text = base.mark_text(fontSize=14, fontWeight='bold').encode(
        text='Count:Q', color=alt.condition(alt.datum.Count > cm.values.max() / 2, alt.value('white'), alt.value('#1C1917')))
    return (rect + text).properties(height=46 * len(cm.index) + 30)


def show_results(res, rows):
    with st.container(horizontal=True):
        st.metric('Accuracy (decided)', pct(res['accuracy']), border=True,
                  help='4-class accuracy on packages that received ACCEPT or REJECT (review cases excluded).')
        st.metric('Macro F1', pct(res.get('macro_f1')), border=True)
        st.metric('False acceptance rate', pct(res['far']), border=True,
                  help='Damaged packages that were ACCEPTED, out of all damaged packages. Lower is better.')
        st.metric('False rejection rate', pct(res['frr']), border=True,
                  help='Intact packages that were REJECTED, out of all intact packages. Lower is better.')
        st.metric('Sent to review', pct(res['review_rate']), border=True,
                  help='Share of package images that received NEEDS REVIEW instead of a decision.')
        if 'invalid_detection_rate' in res:
            st.metric('Non-packages caught', pct(res['invalid_detection_rate']), border=True,
                      help='Images without a package that were flagged as "no package detected".')
    c1, c2 = st.columns([1.1, 1], gap='large')
    with c1.container(border=True):
        st.markdown('**Confusion matrix** :gray[· rows = actual, columns = system output]')
        st.altair_chart(confusion_chart(res['confusion']), width='stretch')
    with c2.container(border=True):
        st.markdown('**Per-class performance** :gray[· decided packages only]')
        if 'per_class' in res:
            st.dataframe(res['per_class'], hide_index=True, width='stretch', column_config={
                c: st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent')
                for c in ('Precision', 'Recall', 'F1-score')})
        st.caption(f"{res['n']} images · {res['n_packages']} packages · {res['n_decided']} decided")
    with st.expander('Every evaluated image', icon=':material/list:'):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


tab_syn, tab_real = st.tabs([':material/auto_awesome: Synthetic held-out set', ':material/photo_camera: Real photos'])

with tab_syn:
    st.markdown(f'Freshly generated packages (seed **{ev.EVAL_SEED}**). The deployed Random Forest + CNN were trained '
                f'on images generated with seed {dd.SEED}, so none of these images were seen during training. They '
                'come from the same generator, so this measures performance on the synthetic domain only.')
    with st.container(horizontal=True, vertical_alignment='bottom'):
        n = st.select_slider('Images per class', [25, 50, 100], value=50, key='ev_n')
        go = st.button('Run synthetic evaluation', type='primary', icon=':material/science:', key='ev_syn_go')
    key = ('syn', n, st.session_state.threshold)
    if go:
        items = [(t, f'synthetic_{t}_{i}', img) for i, (t, img) in enumerate(ev.synthetic_set(n))]
        rows = run_items(items, True, key)
        st.session_state.ev_syn = (key, rows)
    if st.session_state.get('ev_syn') and st.session_state.ev_syn[0] == key:
        rows = st.session_state.ev_syn[1]
        show_results(ev.summarize(rows), rows)
    elif st.session_state.get('ev_syn'):
        st.caption('Settings changed since the last run. Run the evaluation again.')

with tab_real:
    files, desc, independent = ev.real_dataset()
    if independent:
        st.success(f'Dataset: {desc}. Make sure these photos were not used to tune the thresholds.',
                   icon=':material/verified:')
    else:
        st.warning(f'Dataset: {desc}. **This is not an independent test set**: the real-photo thresholds were '
                   'calibrated on these photos, so the numbers are optimistic. For a fair evaluation, put new labelled '
                   'photos in `eval_data/intact/`, `eval_data/crushed/`, `eval_data/torn/`, `eval_data/stained/` and '
                   '`eval_data/not_package/` and reload this page.', icon=':material/warning:')
    compare = st.toggle('Also run the synthetic-trained RF + CNN on these photos (domain-shift check)', value=True,
                        key='ev_cmp')
    go = st.button('Run real-photo evaluation', type='primary', icon=':material/science:', key='ev_real_go')
    key = ('real', desc, st.session_state.threshold, compare)
    if go:
        items = []
        for truth, path in files:
            with open(path, 'rb') as fh:
                try:
                    items.append((truth, os.path.basename(path), ins.load_image(fh.read(), path)))
                except ins.InspectionError as e:
                    st.warning(str(e))
        rows = run_items(items, False, key)
        if compare:
            for row, (truth, name, img) in zip(rows, items):
                a = ui._analyze(img, True)
                row['synthetic_model_argmax'] = ins.CLASSES[int(a['p'].argmax())]
        st.session_state.ev_real = (key, rows)
    if st.session_state.get('ev_real') and st.session_state.ev_real[0] == key:
        rows = st.session_state.ev_real[1]
        show_results(ev.summarize(rows), rows)
        if compare and rows and 'synthetic_model_argmax' in rows[0]:
            pk = [r for r in rows if r['truth'] != 'not_package']
            acc = np.mean([r['synthetic_model_argmax'] == r['truth'] for r in pk]) if pk else None
            intact = [r for r in pk if r['truth'] == 'intact']
            st.info(f"**Domain-shift check:** the Random Forest + CNN trained on synthetic packages classify "
                    f"{pct(acc)} of these real package photos correctly, and call "
                    f"{sum(r['synthetic_model_argmax'] != 'intact' for r in intact)} of {len(intact)} intact boxes "
                    "damaged. This is why real photos are inspected by the CLIP engine.", icon=':material/compare:')
    elif st.session_state.get('ev_real'):
        st.caption('Settings changed since the last run. Run the evaluation again.')
