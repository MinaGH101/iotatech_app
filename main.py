from st_on_hover_tabs import on_hover_tabs
import streamlit as st
import numpy as np 
import pandas as pd
from tools import *
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shap
import os
import joblib
import plotly.graph_objects as go
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from PIL import Image
import plotly.figure_factory as ff
from pygwalker.api.streamlit import StreamlitRenderer
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from st_aggrid import AgGrid
from streamlit_option_menu import option_menu
from tools import *
from plotly.subplots import make_subplots
import importlib.util
from pagess import IOTATECH, trend, exp, Prediction, PredictionR



st.set_page_config(layout="wide")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")


# selected = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal",
#     styles={
#         "container": {"background-color": "#fafafa", "padding" : "15px 600px 15px 15px"},
#         "icon": {"font-size": "15px"}, 
#         "nav-link": {"font-size": "15px", "text-align": "left", "margin-right":"10px"},
#         "nav-link-selected": {"background-color": "#64748b", "font-weight": "400"},
#     }
# )

with st.sidebar:
    tabs = on_hover_tabs(tabName=['IOTATECH', 'Data Trend View', 'Data Exploration', 'Prediction', 'Prediction R&D', 'Help'], 
                         iconName=['psychology', 'data_thresholding', 'settings_applications', 'factory', 'science', 'help'], default_choice=0)


# Function to dynamically import a module
def load_page(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if tabs == "IOTATECH":
    IOTATECH.render_iota()
    # IOTATECH_module = load_page("IOTATECH", "pages0/IOTATECH.py")



elif tabs =='Data Trend View':
    trend.render_trend()
    # trend_module = load_page("Data Trend View", "pages0/Data Trend View.py")


elif tabs =='Data Exploration':
    exp.render_exp()
    # exp_module = load_page("Data Exploration", "pages0/Data Exploration.py")


elif tabs =='Prediction':
    Prediction.render_pred()
    # pred_module = load_page("Prediction", "pages0/Prediction.py")



elif tabs =='Prediction R&D':
    PredictionR.render_predR()
    # pred2_module = load_page("Prediction R&D", "pages0/Prediction R&D.py")
