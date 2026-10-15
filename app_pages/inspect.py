import os

import numpy as np
import streamlit as st

import damage_detector as dd
import inspection as ins
import ui

SOURCES = {'samples': ':material/photo: Sample photos', 'upload': ':material/upload: Upload photo',
           'camera': ':material/photo_camera: Camera', 'synthetic': ':material/auto_awesome: Synthetic generator'}

source = st.segmented_control('Image source', list(SOURCES), format_func=SOURCES.get, default='samples',
                              key='inspect_source') or 'samples'

img, name, synthetic, src = None, 'image', False, 'Upload'
try:
    if source == 'samples':
        files = ui.sample_files()
        st.session_state.setdefault('inspect_sample', None)
        st.caption('Real photos from Wikimedia Commons (licences in samples/ATTRIBUTION.md). The badge is the known label; '
                   'pick a photo to inspect it.')
        with st.container(horizontal=True, gap='small'):          # wraps: ~6 per row on desktop, 2 on a phone
            for f in files:
                with st.container(border=True, gap='small', width=148):
                    st.image(ui.sample_thumb(f), width='stretch')
                    st.markdown(ui.truth_badge(f))
                    selected = st.session_state.inspect_sample == f
                    if st.button(ui.sample_label(f), key=f'pick_{os.path.basename(f)}', width='stretch',
                                 type='primary' if selected else 'secondary'):
                        st.session_state.inspect_sample = f
                        st.rerun()
        f = st.session_state.inspect_sample
        if f:
            with open(f, 'rb') as fh:
                img = ins.load_image(fh.read(), os.path.basename(f))
            name, src = os.path.basename(f), 'Sample'

    elif source == 'upload':
        up = st.file_uploader(f'Upload a package photo (JPG, PNG, WEBP or BMP, up to {ins.MAX_UPLOAD_MB} MB)',
                              type=ins.ALLOWED_EXTENSIONS)
        if up is not None:
            img, name, src = ins.load_image(up.getvalue(), up.name), up.name, 'Upload'

    elif source == 'camera':
        shot = st.camera_input('Take a photo of a package')
        if shot is not None:
            img, name, src = ins.load_image(shot.getvalue(), 'camera photo'), 'camera_photo.jpg', 'Camera'

    else:
        with st.container(horizontal=True, vertical_alignment='bottom'):
            kind = st.pills('Damage type to generate', ['random', *dd.CLASSES], default='random', key='kind')
            if st.button('New package', icon=':material/refresh:', type='primary'):
                st.session_state.seed += 1
        rng = np.random.default_rng(10_000 + st.session_state.seed)
        label = kind if kind and kind != 'random' else dd.CLASSES[int(rng.integers(len(dd.CLASSES)))]
        img = dd.make_package(label, rng)
        name, synthetic, src = f'synthetic_{label}_{st.session_state.seed}.png', True, 'Synthetic'
        st.caption(f'Computer-generated package, known label: **{label}**. Inspected by the Random Forest + CNN '
                   'models that were trained on this kind of synthetic image.')
except ins.InspectionError as e:
    st.error(str(e), icon=':material/broken_image:')
    img = None

if img is None:
    if source == 'samples':
        with st.container(border=True):
            st.markdown('##### :material/touch_app: Pick a sample photo above')
            st.markdown('Each photo is inspected by the real-photo AI engine. Try a crushed box (REJECT), an intact box '
                        '(ACCEPT) and the phone booth or the low-resolution packed box (NEEDS REVIEW).')
    if source in ('upload', 'camera'):
        with st.container(border=True):
            st.markdown('##### :material/add_a_photo: No photo yet')
            st.markdown('Upload or take a photo of a parcel to get an **ACCEPT / REJECT / NEEDS REVIEW** decision, '
                        'the damage type, its location and severity.')
            st.caption('Best results: one package clearly visible, reasonably lit and in focus, filling a good part '
                       'of the frame. Blurry, dark or non-package photos are sent to review instead of guessed.')
    st.stop()

try:
    with st.spinner('Inspecting: segmenting the package and running the AI models...', show_time=True):
        rep = ui.inspect(img, name, src, synthetic=synthetic)
except ins.InspectionError as e:
    st.error(str(e), icon=':material/error:')
    st.stop()

ui.show_report(rep, key='inspect')
ui.show_pipeline(rep)
