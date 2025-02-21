import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from PIL import Image
# import pages
import pathlib
from pathlib import Path
from streamlit_option_menu import option_menu
from st_on_hover_tabs import on_hover_tabs

st.set_page_config(
    page_title="IOTA Tech Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# st.page_link("pages0/Data Trend View.py", label='home', icon=":material/thumb_up:")

pg = st.navigation([st.Page("pages0/IOTATECH.py", title='IOTA TECH', icon=":material/home:"),
                    st.Page("pages0/Data Trend View.py", title='DATA TREND', icon=":material/bar_chart:"),
                    st.Page("pages0/Data Exploration.py", title='DATA EXPLORATION', icon=":material/query_stats:"),
                    st.Page("pages0/Prediction.py", title='PREDICTION', icon=":material/factory:"),
                    st.Page("pages0/Optimization.py", title='OPTIMIZATION', icon=":material/factory:"),
                    st.Page("pages0/Prediction R&D.py", title='LAB PREDICTION', icon=":material/experiment:"),
                    st.Page('pages0/Recommendation.py', title='RECOMMEND', icon=":material/lightbulb:"),
                    st.Page('pages0/Reports.py', title='REPORT', icon=":material/news:"),
                    st.Page('pages0/History.py', title='HISTORY', icon=":material/schedule:"),
                    st.Page('pages0/User Guide.py', title='HELP', icon=":material/help:")], expanded=True)
pg.run()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
