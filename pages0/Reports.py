import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from streamlit_option_menu import option_menu

selected = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"background-color": "#fafafa", "padding" : "15px 600px 15px 15px"},
        "icon": {"font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin-right":"10px"},
        "nav-link-selected": {"background-color": "#64748b", "font-weight": "400"},
    }
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")


st.header("Models report")

# KPI Metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Documents", "10.5K", "⬆ 125")
col2.metric("Annotations", "510", "⬇ 2")
col3.metric("Accuracy", "87.9%", "⬆ 0.1%")
col4.metric("Training Time", "1.5 hours", "⬆ 10 mins")
col5.metric("Processing Time", "3 seconds", "⬇ 0.1 seconds")

# Generate Sample Data
data = pd.DataFrame({
    "x": np.arange(20),
    "a": np.random.randn(20),
    "b": np.random.randn(20),
    "c": np.random.randn(20)
})

c1, c2= st.columns([1, 1])

with c1:
    st.subheader("Model Training")
    fig_bar = px.bar(data, x="x", y=["a", "b"], barmode='group', labels={"value": "Measurement"})
    st.bar_chart(data.iloc[:, 1:], use_container_width=True)

with c2:
    st.subheader("Data Annotation")
    fig_area = px.area(data, x="x", y=["a", "b"], labels={"value": "Measurement"})
    st.area_chart(data.iloc[:, 1:], use_container_width=True)

st.subheader("Data Extraction")
fig_line = px.line(data, x="x", y=["a", "b", "c"], labels={"value": "Measurement"})
st.line_chart(data.iloc[:, 1:], use_container_width=True)

