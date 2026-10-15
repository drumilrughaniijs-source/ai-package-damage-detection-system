"""AI Package Damage Inspector - live demo app.   Run with:  py -3.14 -m streamlit run app.py"""
import streamlit as st
import ui

st.set_page_config(page_title='AI Package Damage Inspector', page_icon=':material/package_2:', layout='wide')
ui.init_state()

page = st.navigation([
    st.Page('app_pages/inspect.py', title='Inspect', icon=':material/search:', default=True),
    st.Page('app_pages/batch.py', title='Batch inspection', icon=':material/photo_library:'),
    st.Page('app_pages/conveyor.py', title='Conveyor simulation', icon=':material/conveyor_belt:'),
    st.Page('app_pages/history.py', title='Inspection history', icon=':material/receipt_long:'),
    st.Page('app_pages/evaluation.py', title='Model evaluation', icon=':material/analytics:'),
    st.Page('app_pages/about.py', title='How it works', icon=':material/school:'),
], position='top')

st.title('AI Package Damage Inspector', icon=':material/package_2:')
st.caption('Photo in → **ACCEPT**, **REJECT** or **NEEDS REVIEW**, with the damage type (crushed, torn, '
           'water-stained), where it is, how severe it is, and why the decision was made.')
stats = ui.sidebar()
ui.load_engines()
page.run()
ui.session_stats(stats)            # after the page, so the counters include this run
