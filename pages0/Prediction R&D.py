import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import joblib


# st.set_page_config(
#     page_title="IOTA Tech Dashboard",
#     page_icon="🏭",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

st.header('📈 Process output prediction')
st.sidebar.write("""
predicts BET (m2/g) based on raw clay analysis and process parameters.
""")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

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

model  = joblib.load('./xgb.pkl')

col1, col2, col3, col4 = st.columns([1.5, 1.5,  1 ,1])

with col1:
    
    f13 = st.select_slider('Acid Type', options=['H2SO4', 'HCl', 'HNO3'])
    f14 = st.slider('Acid Normal', 0.0, 20.0, 2.5, step=0.25)
    f15 = st.slider('Wt. Silica (g)/ V acid (cc)', 0.0, 2.0, 0.05, step=0.01)
    f19 = st.slider('Silica weight (g)', 0.0, 1500.0, 50.0, step=25.0)

    
with col2:
    f16 = st.slider('T(°C)', 80.0, 100.0, 95.0, step=0.25)
    f17 = st.slider('Time (h)', 0.0, 20.0, 3.0, step=0.25)
    f18 = st.slider('Aging Temp', 50.0, 400.0, 180.0, step=10.0)
    f20 = st.slider('Acid volum (cc)', 0.0, 5000.0, 1000.0, step=500.0)


    
with col3:
    f1 = st.number_input("Silica MW", value=360.36, key="Clay MW")
    f2 = st.number_input("initial BET (m2/g)", value=33.3, key="initial BET (m2/g)")
    f3 = st.number_input("d (001) angstrom", value=12.76, key="d (001) angstrom")
    f4 = st.number_input("Al2O3", value=12.7, key="initial Al2O3")
    f5 = st.number_input("Fe2O3", value=3.7, key="initial Fe2O3")
    f6 = st.number_input("CaO", value=5.4, key="initial CaO")

with col4:
    f7 = st.number_input("D90", value=2.8, key="initial MgO")
    f8 = st.number_input("D50", value=1.7, key="initial K2O")
    f9 = st.number_input("D10", value=1.5, key="initial Na2O")
    f10 = st.number_input("Porosity", value=19.2, key="Octa Oxides Sum")
    f11 = st.number_input("Fumed", value=8.6, key="Iintra layer Oxides Sum")
    f12 = st.number_input("Kuartz", value=4.54, key="SiO2/Al2O3")
    

if f13 == 'H2SO4':
    f13_ = 98.08
elif f13 == 'HCl':
    f13_ = 36.46
else:
    f13_ = 63.1

X1 = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9,
                f10, f11, f12, f13_, f14, f15, f16, f17, f18], dtype='float64').reshape(1, 18)

y_pred1 = model.predict(X1)

st.subheader(f'Predicted DLS: {np.round(y_pred1, 0)} ± 10 (nm)')


# model1  = joblib.load('./octa_gb.pkl')
# X2 = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9,
#                 f10, f11, f12, f13_, f14, f15, f19, f20, f16, f17, f18], dtype='float64').reshape(1, 20)
# y_pred2 = model1.predict(X2)
# st.subheader(f'Sum of octahedral metals: {np.round(y_pred2, 0)} ± 3 (Wt. %)')