import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import joblib


st.set_page_config(
    page_title="Kharazmi Activeclay Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header('📝 Oprator FeedBack')

st.markdown("""
        <style>
               .css-12oz5g7 {
                    padding-top: 1rem;
                    padding-bottom: 10rem;
                    padding-left: 2rem;
                    padding-right: 0rem;
                }
               .css-12oz5g7 {
                    padding-top: 2rem;
                    padding-right: 1rem;
                    padding-bottom: 2rem;
                    padding-left: 2rem;
                }
                .css-18e3th9 {
                    padding-top: 2rem;
                    padding-right: 4rem;
                    padding-bottom: 2rem;
                    padding-left: 4rem;
                }
        </style>
        """, unsafe_allow_html=True)



col11, col22 = st.columns([6, 2])
with col11:
    with st.form("my_form", clear_on_submit=False):
        
        col1, col2 = st.columns([2 ,4], gap='medium')
        with col1:
            st.date_input(label='date of action')
            st.time_input('time of action')
            st.text_input("operatior's name")
            st.text_input("operator's ID")
            st.selectbox('action type', options=['Preventing', 'improving'])



        with col2:


            action = st.text_area(label='Action')
            cause = st.text_area(label='Cause')

            reset_button = st.form_submit_button("Submit", type='primary')


with col22:
    st.subheader('feedback history', divider=True)
