import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from PIL import Image
# import pages
import pathlib
from pathlib import Path

st.set_page_config(
    page_title="IOTA Tech Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([st.Page("pages0/IOTATECH.py", icon='🏭'),
                    st.Page("pages0/Data Trend View.py", icon='📈'),
                    st.Page("pages0/Data Exploration.py", icon='🔎'),
                    st.Page("pages0/Prediction.py", icon='⚙️'),
                    st.Page("pages0/Prediction R&D.py", icon='🗳'),
                    st.Page('pages0/Recommendation.py', icon='🔧'),
                    st.Page('pages0/User Guide.py', icon='📄')], expanded=True)
pg.run()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
