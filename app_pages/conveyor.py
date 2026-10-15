import time

import numpy as np
import pandas as pd
import streamlit as st

import damage_detector as dd
import inspection as ins
import ui

LANES = {'ACCEPT': ('Accept lane', ':material/check_circle:', 'green'),
         'REJECT': ('Reject lane', ':material/cancel:', 'red'),
         'NEEDS REVIEW': ('Manual review lane', ':material/pending_actions:', 'orange')}
STAGES = ['Enters inspection area', 'AI inspection', 'Decision', 'Routed to lane']

st.info('**Simulation.** No physical conveyor or industrial camera is connected: packages are taken from the '
        'sample photos or the synthetic generator, then inspected with the real models and decision rules.',
        icon=':material/science:')

with st.container(horizontal=True, vertical_alignment='bottom'):
    kind = st.segmented_control('Packages', ['Real sample photos', 'Synthetic packages'], default='Real sample photos',
                                key='belt_kind') or 'Real sample photos'
    n = st.slider('Packages on the belt', 4, 16, 8, step=4)
    run = st.button('Run conveyor', icon=':material/play_arrow:', type='primary')

if run:
    rng = np.random.default_rng()
    items = []
    if kind == 'Synthetic packages':
        for i, t in enumerate(dd.CLASSES[j] for j in rng.integers(len(dd.CLASSES), size=n)):
            items.append((t, dd.make_package(t, rng), True, f'belt_{i + 1}_synthetic.png'))
    else:
        files = ui.sample_files()
        for i, f in enumerate(rng.choice(files, size=n, replace=n > len(files))):
            with open(f, 'rb') as fh:
                img = ins.load_image(fh.read(), f)
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()                  # mirrored copies so repeated photos are not identical
            items.append((ui.truth_of(f), img, False, f'belt_{i + 1}_{ui.sample_label(f).lower().replace(" ", "_")}'))

    live = st.empty()
    belt = []
    for i, (truth, img, synth, name) in enumerate(items):
        def stage(k, r=None, img=img, i=i):
            with live.container(border=True):
                st.markdown(f'**Package {i + 1} of {len(items)}** :gray[· simulated conveyor]')
                st.markdown('  →  '.join(f'**:orange[{s}]**' if j == k else (f':gray[{s}]' if j > k else s)
                                         for j, s in enumerate(STAGES)))
                c1, c2 = st.columns([1, 2], vertical_alignment='center')
                c1.image(ui.rgb(r['annotated'] if r else img), width='stretch')
                with c2:
                    if r:
                        ui.decision_banner(r)
                        if k == 3:
                            st.markdown(f"Routed to the **{LANES[r['decision']][0].lower()}**.")
                    st.progress((i + (k + 1) / len(STAGES)) / len(items))
        stage(0)
        time.sleep(0.25)
        stage(1)
        try:
            r = ui.inspect(img, name, 'Conveyor (simulated)', synthetic=synth)
        except ins.InspectionError as e:
            st.error(f'Package {i + 1}: {e}', icon=':material/error:')
            continue
        stage(2, r)
        time.sleep(0.35)
        stage(3, r)
        time.sleep(0.25)
        belt.append((truth, ui.slim(r)))
    live.empty()
    st.session_state.belt = belt

belt = st.session_state.get('belt')
if not belt:
    with st.container(border=True):
        st.markdown('##### :material/conveyor_belt: Belt is idle')
        st.markdown('Press **Run conveyor**: each package enters the inspection area, is inspected, receives a '
                    'decision and is routed to the accept, reject or manual-review lane.')
    st.stop()

counts = {d: sum(r['decision'] == d for _, r in belt) for d in ins.DECISIONS}
decided = [(t, r) for t, r in belt if r['decision'] != 'NEEDS REVIEW']
correct = sum(ins.predicted_label(r) == t for t, r in decided)
with st.container(horizontal=True):
    st.metric('Inspected', len(belt), border=True)
    st.metric('Accepted', counts['ACCEPT'], border=True)
    st.metric('Rejected', counts['REJECT'], border=True)
    st.metric('Manual review', counts['NEEDS REVIEW'], border=True)
    st.metric('Decided correctly', f'{correct}/{len(decided)}' if decided else '-', border=True,
              help='Compared with the known label of each sample / generated package. Review cases are excluded.')

for col, (decision, (title, icon, colour)) in zip(st.columns(3, gap='medium'), LANES.items()):
    with col:
        group = [(t, r) for t, r in belt if r['decision'] == decision]
        st.subheader(f':{colour}[{icon}] {title} ({len(group)})')
        if not group:
            st.caption('Empty')
        for row in range(0, len(group), 2):
            for c, (t, r) in zip(st.columns(2), group[row:row + 2]):
                with c.container(border=True):
                    st.image(ui.rgb(r['annotated']), width='stretch')
                    label = ins.predicted_label(r)
                    mark = ('' if decision == 'NEEDS REVIEW' else
                            ':green[:material/check:] ' if label == t else ':red[:material/close:] ')
                    st.caption(f"{mark}{r['damage_type'] or ('-' if decision != 'ACCEPT' else 'intact')} · "
                               f"known: {t}")

with st.expander('Belt log', icon=':material/list:'):
    st.dataframe(pd.DataFrame([{**ui.result_row(r), 'Known label': t} for t, r in belt]), hide_index=True,
                 width='stretch')
