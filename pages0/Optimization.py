import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from PIL import Image
from tools import *
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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

# image11 = Image.open('images/hmi.png')
# st.image(image11, width=1000)


with open("images/cement_hmi.svg", "r") as svg_file:
    svg_content = svg_file.read()


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
                .main-svg {
                    height: 510px important!;
                    }
                }
        </style>
        """, unsafe_allow_html=True)

# DATA PREPARATION  #############################################################################

if 'M' not in st.session_state:
    st.session_state.M = pd.read_csv('./data/ais_metal_total_5.0_lag_ordered_new.csv')
    st.session_state.temp_params0 = pd.read_excel('./data/final_metal_features.xlsx')['Parameter Name'].to_list()

M = st.session_state.M
temp_params = st.session_state.temp_params0

if 'X0' not in st.session_state:
    X0, d = data_preparation1(M, temp_params)
    X_train0, y_train0, X_test0, y_test0, Fe_total, X0, y0 = data_preparation2(X0, 'MD', d, ts = 2000)
    st.session_state.X0 = X0
    st.session_state.y0 = y0

X0 = st.session_state.X0
y0 = st.session_state.y0    
Data0 = pd.concat([y0, M['%C'], M['WT_X641'], X0], axis=1)
Date_time0 = M['date_time']
date0 = pd.DataFrame()
date0['date'] = pd.to_datetime(M['date_time']).dt.date
sd10 = date0['date'].iloc[0]
ed10 = date0['date'].iloc[-1]


if 'D' not in st.session_state:
    st.session_state.D = pd.read_csv('./data/pump_data.csv')
    st.session_state.D = st.session_state.D.dropna(how='any', axis=0)
    st.session_state.temp_params = st.session_state.D.columns.to_list()
    st.session_state.date_time = pd.to_datetime(st.session_state.D['date'])

D = st.session_state.D
temp_params = st.session_state.temp_params

if 'X' not in st.session_state:
    st.session_state.X = D.iloc[:, 1:43]
    st.session_state.y = D[['machine_status']]
    st.session_state_date_time = pd.to_datetime(D['date'])

X = st.session_state.X
y = st.session_state.y    
Data = pd.concat([y, X], axis=1)
Date_time = st.session_state.date_time
date = pd.DataFrame()
date['date'] = pd.to_datetime(Date_time).dt.date
sd1 = date['date'].iloc[0]
ed1 = date['date'].iloc[-1]


# y_ = st.selectbox('Target Column', options=Data.columns)
# y = Data[y_]

#############################################################################################
st.header('Process prediction')



st.markdown(f"<div>{svg_content}</div>", unsafe_allow_html=True)



##############################################################################################

col11, col22, col33 = st.columns([1, 4, 1])

with col11:
    
    lag = st.radio(options=['1 Hours 🕒', '2 Hours 🕔', '3 Hours 🕗'], label='⌛️ Predictions for the next ...')

    if lag == '1 Hours 🕒':

        MD_now = 93.5
        C_now = 1.8
        PR_now = 115

    elif lag == '2 Hours 🕔':
        MD_now = 93.1
        C_now = 1.9
        PR_now = 112

    elif lag == '3 Hours 🕗':
        MD_now = 92.8
        C_now = 2.1
        PR_now = 105

    MD_last = Data0['MD'].iloc[-2]
    delta = MD_now - MD_last
    st.metric(label="Process Yield (%)", value=f"{MD_now} %", delta=f"{delta} %")

    C_last = Data0['%C'].iloc[-2]
    delta = C_now - C_last
    st.metric(label="Out Flow", value=f"{C_now}", delta=f"{delta}")

    PR_last = Data0['WT_X641'].iloc[-5]
    delta = PR_now - PR_last
    st.metric(label="Power Usage", value=f"{PR_now}", delta=f"{delta}")


  
with col22:

    X = Date_time0.iloc[-24:]
    # print(X)
    y1=Data0['MD'].replace(0, 91).iloc[-24:]
    y2=Data0['%C'].iloc[-24:]
    y3=Data0['WT_X641'].iloc[-24:]

    y1lim = [89.5, 95]
    y2lim = [0.5, 3]
    y3lim = [100, 120]

    T = pd.concat([X, y1, y2, y3], axis=1)

    # st.area_chart(T, x='date_time', y='MD')

    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.2)
    fig.add_trace(
        go.Scatter(x=X, y=y1, fill="tonexty"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=X, y=y2, fill="tonexty"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=X, y=y3, fill="tonexty"),
        row=3, col=1
    )
    fig.update_layout(autosize=True,
                  height=600,
                 )
    fig.update_layout(yaxis1=dict(range=[y1lim[0],y1lim[1]]),
                      yaxis2=dict(range=[y2lim[0],y2lim[1]]),
                      yaxis3=dict(range=[y3lim[0],y3lim[1]]))


    # fig = px.area(T, x="date_time", y="WT_X641", range_y=(100, 120), height=250)
    st.plotly_chart(fig, use_container_width=True, key='MD_trend')


with col33:

    st.container(height=75, border=False)

    MD_now = Data0['MD'].iloc[-1]
    MD_last = Data0['MD'].iloc[-2]
    delta = MD_now - MD_last
    st.metric(label="Process Yield (%)", value=f"{MD_now} %", delta=f"{delta} %")
    st.divider()

    C_now = Data0['%C'].iloc[-1]
    C_last = Data0['%C'].iloc[-2]
    delta = C_now - C_last
    st.metric(label="Out Flow", value=f"{C_now}", delta=f"{delta}")
    st.divider()

    PR_now = Data0['WT_X641'].iloc[-4]
    PR_last = Data0['WT_X641'].iloc[-5]
    delta = PR_now - PR_last
    st.metric(label="Power Usage", value=f"{PR_now}", delta=f"{delta}")

# image = Image.open('images/hmi2.jpg')
# st.image(image)

""" _______________________________________________________________________________________________"""
# model  = joblib.load('./xgb.pkl')

options = st.multiselect(
"Parameters",
list(Data.columns),
list(Data.columns)[:20])

col1, col2, col3, col4, col5 = st.columns([1, 1, 1 ,1, 1])
features = []
n = int(len(options)/5)
e = int(len(options)//5)


use_default = st.toggle("Use Default Values")
for tag in options[:n]:
    with col1:
        tag = st.number_input(tag, value=Data[tag].mode().values[0], key=tag)
        features.append(tag)

for tag in options[n:2*n]:
    with col2:
        tag = st.number_input(tag, value=Data[tag].mode().values[0], key=tag)
        features.append(tag)

for tag in options[2*n:3*n]:
    with col3:
        tag = st.number_input(tag, value=Data[tag].mode().values[0], key=tag)
        features.append(tag)

for tag in options[3*n:4*n]:
    with col4:
        tag = st.number_input(tag, value=Data[tag].mode().values[0], key=tag)
        features.append(tag)

for tag in options[4*n:]:
    with col5:
        tag = st.number_input(tag, value=Data[tag].mode().values[0], key=tag)
        features.append(tag)




X_in = np.array(features)

dd = st.data_editor(Data[options].iloc[-8:])

