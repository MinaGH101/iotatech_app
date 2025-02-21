import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from inverse import X_Recommender
from streamlit_option_menu import option_menu

# st.set_page_config(
#     page_title="Kharazmi Activeclay Dashboard",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

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


st.header('🔧 Recommendation Mode')

# st.sidebar.write("""
# recommends parameters to reach a specific BET (m2/g).
# Based on the practical experiments, it can be observed that the optimal values are relatively consistent across different clay types, and there is little dependency.
# """)

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

model  = joblib.load('./data/xgb.pkl')

col1, col2, col3, col4 = st.columns([1, 1,  1 ,1])

with col1:
    f1 = st.number_input("Clay MW", value=360.36, key="Clay MW")
    f2 = st.number_input("initial BET (m2/g)", value=33.3, key="initial BET (m2/g)")
    f3 = st.number_input("d (001) angstrom", value=12.76, key="d (001) angstrom")
with col2:
    f4 = st.number_input("initial Al2O3", value=12.7, key="initial Al2O3")
    f5 = st.number_input("initial Fe2O3", value=3.7, key="initial Fe2O3")
    f6 = st.number_input("initial CaO", value=5.4, key="initial CaO")
with col3:
    f7 = st.number_input("initial MgO", value=2.8, key="initial MgO")
    f8 = st.number_input("initial K2O", value=1.7, key="initial K2O")
    f9 = st.number_input("initial Na2O", value=1.5, key="initial Na2O")
with col4:
    sum = f4+f5+f7
    f10 = st.number_input("Octa Oxides Sum", value=sum, key="Octa Oxides Sum")
    f11 = st.number_input("Iintra layer Oxides Sum", value=8.6, key="Iintra layer Oxides Sum")
    f12 = st.number_input("SiO2/Al2O3", value=4.54, key="SiO2/Al2O3")
    
# f13 = st.select_slider('Acid Type', options=['H2SO4', 'HCl', 'HNO3'])

f13 = st.radio('Acid Type', options=['H2SO4', 'HCl', 'HNO3'], horizontal=True)
if f13 == 'H2SO4':
    f13_ = 98.08
elif f13 == 'HCl':
    f13_ = 36.46
else:
    f13_ = 63.1
    
clay = [f1, f2, f3, f4, f5, f6, f7, f8, f9,
                f10, f11, f12, f13_]

y_max, X_pred = X_Recommender(clay, model)
X = pd.DataFrame(np.array(X_pred).reshape(-1, 18)[:, 13:],
                 columns=['Acid Normal', 'wt clay/ V acid (g/cc)',
                            'T (°C)', 'Time (h)', 'Highest seen Temp (°C)'])
st.write('Here are approximated optimal vlaues:')
X = pd.DataFrame(X).round(3)
st.dataframe(X.drop_duplicates())
c11 , c22 = st.columns(2)
with c11:
    st.subheader(f'maximun BET : {np.round(float(y_max), 1)} (m2/g)')
#     st.subheader(f'Acid Normal : {np.round(X_pred[:, 13], 1)} (N)')
#     st.subheader(f'wt clay/ V acid : {np.round(X_pred[:, 14], 1)} (g/cc)')
    
# with c22:
#     st.subheader(f'T : {np.round(X_pred[:, 15], 1)} ')
#     st.subheader(f'Time : {np.round(X_pred[:, 16], 1)} (h)')
#     st.subheader(f'highest seen Temp : {np.round(X_pred[:, 17], 1)} (°C)')
