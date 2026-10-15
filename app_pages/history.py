import json

import cv2
import streamlit as st

import inspection as ins
import storage
import ui

try:
    df = storage.load()
except Exception as e:
    st.error(f'The inspection history could not be read ({e}).', icon=':material/error:')
    st.stop()

if df.empty:
    with st.container(border=True):
        st.markdown('##### :material/receipt_long: No inspections stored yet')
        st.markdown('Every inspection (single, batch or conveyor) is saved here automatically, with its decision, '
                    'model outputs, processing time and annotated image. It stays available after restarting the app.')
    st.stop()

with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment='bottom'):
        dec = st.pills('Decision', ins.DECISIONS, selection_mode='multi', default=ins.DECISIONS, key='h_dec')
        types = st.multiselect('Damage type', ins.CLASSES[1:], key='h_type', placeholder='All types')
        sources = st.multiselect('Source', sorted(df.source.dropna().unique()), key='h_src', placeholder='All sources')
        query = st.text_input('Search ID or file name', key='h_q', placeholder='e.g. fedex')

f = df[df.decision.isin(dec or [])]
if types:
    f = f[f.damage_type.isin(types)]
if sources:
    f = f[f.source.isin(sources)]
if query:
    q = query.strip().lower()
    f = f[f.id.str.lower().str.contains(q, regex=False) | f.filename.str.lower().str.contains(q, regex=False)]

with st.container(horizontal=True):
    st.metric('Shown', len(f), border=True)
    st.metric('Accepted', int((f.decision == 'ACCEPT').sum()), border=True)
    st.metric('Rejected', int((f.decision == 'REJECT').sum()), border=True)
    st.metric('Needs review', int((f.decision == 'NEEDS REVIEW').sum()), border=True)
    st.metric('Avg. processing', f'{f.processing_ms.mean() / 1000:.2f} s' if len(f) else '-', border=True)

left, right = st.columns(2, gap='large')
with left.container(border=True):
    st.markdown('**Decisions**')
    st.altair_chart(ui.count_chart(f.decision.value_counts().reindex(ins.DECISIONS, fill_value=0), ui.DECISION_HEX),
                    width='stretch')
with right.container(border=True):
    st.markdown('**Damage types among rejected packages**')
    rej = f[f.decision == 'REJECT']
    if len(rej):
        st.altair_chart(ui.count_chart(rej.damage_type.value_counts(), {k: v for k, v in ui.CLASS_COLOURS.items()
                                                                        if k != 'intact'}), width='stretch')
    else:
        st.caption('No rejected packages in this selection.')

st.caption('Select a row to see the full inspection.')
f = f.assign(timestamp=f.timestamp.str.replace('T', ' '), damage_type=f.damage_type.fillna('-'),
             severity=f.severity.fillna('-'))
table = f.rename(columns={'id': 'ID', 'timestamp': 'Time', 'filename': 'File', 'source': 'Source',
                          'engine': 'Engine', 'decision': 'Decision', 'damage_type': 'Damage type',
                          'severity': 'Severity', 'p_intact': 'P(intact)', 'top_class': 'Top class',
                          'top_class_probability': 'Top class prob.', 'threshold': 'Threshold',
                          'damaged_area_pct': 'Damaged area %', 'processing_ms': 'Processing ms'})
event = st.dataframe(table, hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row',
                     key='h_table', column_config={
                         'Time': st.column_config.TextColumn(width='medium'),
                         'P(intact)': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent'),
                         'Top class prob.': st.column_config.NumberColumn(format='%.2f'),
                         'Damaged area %': st.column_config.NumberColumn(format='%.0f%%'),
                         'Processing ms': st.column_config.NumberColumn(format='%.0f')})

with st.container(horizontal=True):
    st.download_button('Export shown rows (CSV)', table.to_csv(index=False), 'inspection_history.csv', 'text/csv',
                       icon=':material/download:')
    with st.popover('Clear history', icon=':material/delete:'):
        st.markdown('Delete **all** stored inspections and their images? This cannot be undone.')
        if st.button('Yes, delete everything', type='primary'):
            storage.clear()
            st.session_state.session = []
            st.rerun()

rows = event.selection.rows
if rows:
    rid = table.iloc[rows[0]]['ID']
    item = storage.get(rid)
    if item is None:
        st.warning('This inspection is no longer stored.', icon=':material/warning:')
        st.stop()
    rep, annotated, original = item
    with st.container(border=True):
        colour = ui.DECISION_COLOUR[rep['decision']]
        st.markdown(f"#### :{colour}[{rep['decision']}] · {rep['id']}")
        st.caption(f"{rep['timestamp'].replace('T', ' ')} · {rep['filename']} · {rep['source']} · {rep['model']}")
        c1, c2, c3 = st.columns([1, 1, 1.1], gap='medium')
        if original is not None:
            c1.image(ui.rgb(original), caption='Original', width='stretch')
        if annotated is not None:
            c2.image(ui.rgb(annotated), caption='Annotated result', width='stretch')
        with c3:
            with st.container(horizontal=True):
                st.metric('P(intact)', f"{rep['p_intact']:.0%}", border=True)
                st.metric('Damage type', (rep['damage_type'] or '-').capitalize(), border=True)
            with st.container(horizontal=True):
                st.metric('Severity', rep['severity'] or '-', border=True)
                st.metric('Threshold', f"{rep['threshold']:.2f}", border=True)
            st.markdown('\n'.join(f'- {line}' for line in rep['explanation']))
            if annotated is not None:
                ok, png = cv2.imencode('.png', annotated)
                st.download_button('Annotated image', png.tobytes(), f"{rid}_annotated.png", 'image/png',
                                   icon=':material/download:')
            st.download_button('JSON report', json.dumps(rep, indent=2), f'{rid}_report.json', 'application/json',
                               icon=':material/data_object:')
