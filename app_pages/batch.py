import io
import json
import os
import zipfile

import pandas as pd
import streamlit as st

import inspection as ins
import ui

MAX_FILES = 40

st.markdown('Inspect a whole delivery at once. Every photo goes through exactly the same validation, models and '
            'decision rules as a single inspection, and is saved to the inspection history.')

with st.container(border=True):
    ups = st.file_uploader(f'Package photos (up to {MAX_FILES}, JPG / PNG / WEBP / BMP, {ins.MAX_UPLOAD_MB} MB each)',
                           type=ins.ALLOWED_EXTENSIONS, accept_multiple_files=True, key='batch_files')
    with st.container(horizontal=True):
        start = st.button(f'Inspect {len(ups)} uploaded photo{"s" if len(ups) != 1 else ""}' if ups else 'Inspect uploads',
                          type='primary', icon=':material/play_arrow:', disabled=not ups)
        samples = st.button('Inspect all sample photos', icon=':material/photo_library:')

if start or samples:
    if samples:
        items = []
        for f in ui.sample_files():
            with open(f, 'rb') as fh:
                items.append((os.path.basename(f), fh.read()))
    else:
        items = [(u.name, u.getvalue()) for u in ups]
    skipped = [(n, f'Only the first {MAX_FILES} files are inspected.') for n, _ in items[MAX_FILES:]]
    items = items[:MAX_FILES]

    results, failed = [], list(skipped)
    with st.status(f'Inspecting {len(items)} photos...', expanded=True) as status:
        bar = st.progress(0.0)
        for i, (name, data) in enumerate(items):
            bar.progress(i / len(items), text=f'{i + 1}/{len(items)} · {name}')
            try:
                results.append(ui.slim(ui.inspect(ins.load_image(data, name), name, 'Batch')))
            except ins.InspectionError as e:
                failed.append((name, str(e)))
        bar.progress(1.0, text='Done')
        status.update(label=f'Inspected {len(results)} of {len(items)} photos'
                            + (f' · {len(failed)} failed or skipped' if failed else ''),
                      state='error' if failed and not results else 'complete', expanded=False)
    st.session_state.batch = {'results': results, 'failed': failed}

batch = st.session_state.get('batch')
if not batch:
    with st.container(border=True):
        st.markdown('##### :material/photo_library: No batch yet')
        st.markdown('Upload several photos and press **Inspect**, or try **Inspect all sample photos**.')
    st.stop()

results, failed = batch['results'], batch['failed']
counts = {d: sum(r['decision'] == d for r in results) for d in ins.DECISIONS}
with st.container(horizontal=True):
    st.metric('Inspected', len(results), border=True)
    st.metric('Accepted', counts['ACCEPT'], border=True)
    st.metric('Rejected', counts['REJECT'], border=True)
    st.metric('Needs review', counts['NEEDS REVIEW'], border=True)
    st.metric('Failed / skipped', len(failed), border=True)

if failed:
    with st.expander(f'{len(failed)} file(s) could not be inspected', icon=':material/error:', expanded=True):
        st.dataframe(pd.DataFrame(failed, columns=['File', 'Reason']), hide_index=True, width='stretch')

if results:
    df = pd.DataFrame([ui.result_row(r) for r in results])
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
        for r in results:
            z.writestr(f"{r['id']}_{ui.safe_name(r['filename'])}_annotated.png", ins.annotated_png(r))
    with st.container(horizontal=True):
        st.download_button('Results (CSV)', df.to_csv(index=False), 'batch_results.csv', 'text/csv',
                           icon=':material/table_view:')
        st.download_button('Annotated images (ZIP)', zbuf.getvalue(), 'batch_annotated.zip', 'application/zip',
                           icon=':material/folder_zip:')
        st.download_button('Reports (JSON)', json.dumps([ins.export_dict(r) for r in results], indent=2),
                           'batch_reports.json', 'application/json', icon=':material/data_object:')

    show = st.pills('Show', ins.DECISIONS, selection_mode='multi', default=ins.DECISIONS, key='batch_filter')
    shown = [r for r in results if r['decision'] in (show or [])]
    st.dataframe(df[df.Decision.isin(show or [])], hide_index=True, width='stretch', column_config={
        'P(intact)': st.column_config.ProgressColumn(min_value=0, max_value=1, format='percent'),
        'Damaged area %': st.column_config.NumberColumn(format='%.0f%%')})

    for row in range(0, len(shown), 4):
        for col, r in zip(st.columns(4), shown[row:row + 4]):
            with col.container(border=True):
                st.image(ui.rgb(r['annotated']), width='stretch')
                what = f" · {r['damage_type']}" if r['decision'] == 'REJECT' else ''
                st.markdown(f"{ui.decision_badge(r)}{what}  \n:gray[{r['filename']}]")
                st.caption(f"P(intact) {r['p_intact']:.0%} · severity {r['severity'] or '-'} · "
                           f"{r['processing_ms'] / 1000:.1f} s")
                st.download_button('Annotated', ins.annotated_png(r), f"{r['id']}_annotated.png", 'image/png',
                                   icon=':material/download:', key=f"dl_{r['id']}", width='stretch')
