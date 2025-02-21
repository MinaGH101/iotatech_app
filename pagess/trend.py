from tools import *
def render_trend():

    import streamlit as st 
    import numpy as np 
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.impute import KNNImputer
    from PIL import Image
    import plotly.figure_factory as ff
    
    import plotly.express as px
    from pygwalker.api.streamlit import StreamlitRenderer
    from scipy.signal import savgol_filter
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from st_aggrid import AgGrid
    from streamlit_option_menu import option_menu



    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")

    st.sidebar.markdown(
        f"""
        <style>
        div.css-6qob1r {{
            background-image: linear-gradient(20deg, rgba(129,235,255,0.4) 0%, rgba(255,255,255,0) 48%);

            padding: 0;
            margin: 0;
        }}

        div.st-emotion-cache-vft1hk {{
            align-items: center;
            align-content: center;
        }}
        </style>

        """,
        unsafe_allow_html=True,
        )
    
    selected = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
            "container": {"background-color": "#FFFFFF", "padding" : "15px 600px 15px 15px"},
            "icon": {"font-size": "15px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin-right":"10px"},
            "nav-link-selected": {"background-color": "#64748b", "font-weight": "400"},
        }
    )

    st.header('Data Trend View')
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

    tab0, tab1, tab2, tab3, tab4 = st.tabs(['Data Overview', 'X_Y Plot', 'X_Y_Z Plot', 'X_Time Plot', 'Hist Plot'])


    with tab0: 
        # df = pd.read_excel(Data) 
        p = StreamlitRenderer(Data)
        p.explorer()

        filt = st.checkbox('Filter data', key='filt')
        if filt:
            AgGrid(Data)




    with tab1:

        plot_number = st.number_input('number of plots to compare ...', value=1, key="plot_number1")
        st.divider()

        for i in range(plot_number):
            st.subheader(f'plot number {i+1}', divider=True)
            col0, col1, col2 = st.columns([2, 2, 4])

            with col0:
                    X_name = st.selectbox(
                        'Select X-axis',
                        (Data.columns), key=f'X_name{i}')

                    y_name = st.selectbox(
                        'Select Y-axis',
                        (Data.columns), key=f'y_name{i}')
                    
                    color = st.selectbox(
                        'Select Color Param',
                        (Data.columns), key=f'z_name{i}')

                    # color = st.selectbox(
                    #     'Select a color',
                    #     ('blue', 'red', 'forestgreen', 'purple', 'blueviolet', 'aqua'))

            with col1:
                
                with st.form(f"my_form{i}", clear_on_submit=False):
                    X_min = st.number_input("X_min", value=Data[X_name].min(), key=f"X_min{i}")
                    X_max = st.number_input("X_max", value=Data[X_name].max(), key=f"X_max{i}")
                    y_min = st.number_input("y_min", value=Data[y_name].min(), key=f"y_min{i}")
                    y_max = st.number_input("y_max", value=Data[y_name].max(), key=f"y_max{i}")
                    
                    if X_min:
                        Xmin = X_min
                    else:
                        Xmin = X.min()
                        
                    if X_max:
                        Xmax = X_max
                    else:
                        Xmax = X.max()
                    
                #### 
                    if y_min:
                        ymin = y_min
                    else:
                        ymin = y.min()
                        
                    if y_max:
                        ymax = y_max
                    else:
                        ymax = y.max()

                    reset_button = st.form_submit_button("Set / Reset")

            with col2:

                fig = px.scatter(Data, x=X_name, y=y_name, range_x=[Xmin, Xmax], range_y=[ymin, ymax], color=color)
                st.plotly_chart(fig, use_container_width=True, key=f'sc_plot{i}')



    with tab2:

        plot_number = st.number_input('number of plots to compare ...', value=1, key="plot_number2")
        st.divider()

        for i in range(plot_number):
            
            st.subheader(f'plot number {i+1}', divider=True)

            col0, col1, col2 = st.columns([1.5, 1.5, 5])

            with col0:
                    X_name = st.selectbox(
                        'Select X-axis',
                        (Data.columns), key=f"X_name{i}{i}")

                    y_name = st.selectbox(
                        'Select Y-axis',
                        (Data.columns), key=f"y_name{i}{i}")
                    
                    z_name = st.selectbox(
                        'Select Z-axis',
                        (Data.columns), key=f"z_name{i}{i}")
                    
                    color = st.selectbox(
                        'Select Color Param',
                        (Data.columns), key=f"color{i}{i}")
                    
                    start_date = st.date_input("Start_Date", key=f'sd2{i}{i}')
                    end_date = st.date_input("End_Date", key=f'ed2{i}{i}')
                    
                    X, y, z = Data[X_name], Data[y_name], Data[z_name]

                    
            with col1:
                
                with st.form(f"my_form1{i}{i}", clear_on_submit=False):
                    X_min = st.number_input("X_min", value=Data[X_name].min(), key=f"X_min2{i}{i}")
                    X_max = st.number_input("X_max", value=Data[X_name].max(), key=f"X_max2{i}{i}")
                    y_min = st.number_input("y_min", value=Data[y_name].min(), key=f"y_min2{i}{i}")
                    y_max = st.number_input("y_max", value=Data[y_name].max(), key=f"y_max2{i}{i}")
                    z_min = st.number_input("z_min", value=Data[z_name].min(), key=f"z_min2{i}{i}")
                    z_max = st.number_input("z_max", value=Data[z_name].max(), key=f"z_max2{i}{i}")
                    
                    if X_min:
                        Xmin = X_min
                    else:
                        Xmin = X.min()
                        
                    if X_max:
                        Xmax = X_max
                    else:
                        Xmax = X.max()
                    
                #### 
                    if y_min:
                        ymin = y_min
                    else:
                        ymin = y.min()
                        
                    if y_max:
                        ymax = y_max
                    else:
                        ymax = y.max()

                #####
                    if z_min:
                        zmin = z_min
                    else:
                        zmin = z.min()
                        
                    if z_max:
                        zmax = z_max
                    else:
                        zmax = z.max()

                    reset_button = st.form_submit_button("Set / Reset")

            with col2:

                fig = px.scatter_3d(Data, x=X_name, y=y_name, z=z_name, 
                                    range_x=[Xmin, Xmax], 
                                    range_y=[ymin, ymax], 
                                    range_z=[zmin, zmax], 
                                    color=color)
                st.plotly_chart(fig, use_container_width=True, key=f'3d_plot{i}')



    with tab3:

        plot_number = st.number_input('number of plots to compare ...', value=1, key="plot_number3")
        st.divider()

        for i in range(plot_number):

            st.subheader(f'plot number {i+1}', divider=True)
            col0, col1 = st.columns([2, 6])

            with col0:

                    y_name = st.selectbox(
                        'Select Y-axis',
                        (Data.columns), key=f"y_name3{i}{i}{i}")
                    
                    start_date = st.date_input("Start_Date", key=f'sd3{i}{i}{i}')
                    end_date = st.date_input("End_Date", key=f'ed3{i}{i}{i}')


                    with st.form(f"my_form3{i}{i}{i}", clear_on_submit=False):
                        y_min = st.number_input("y_min", value=Data[y_name].min(), key=f"y_min3{i}{i}{i}")
                        y_max = st.number_input("y_max", value=Data[y_name].max(), key=f"y_max3{i}{i}{i}")
                        

                        if y_min:
                            ymin = y_min
                        else:
                            ymin = y.min()
                            
                        if y_max:
                            ymax = y_max
                        else:
                            ymax = y.max()

                    #####
                        if (start_date < sd1) | (start_date > ed1):
                            sd = sd1
                        else:
                            sd = start_date
                            
                        if (end_date < sd1) | (end_date > ed1):
                            ed = ed1
                        else:
                            ed = end_date

                    #####

                        reset_button = st.form_submit_button("Set / Reset")
                    
                    
            with col1:

                col11, col12, col13 = st.columns([3, 5, 5])
                with col11:
                    smooth = st.checkbox("Smooth", key=f'smooth{i}')

                start_idx = date.index[date['date'] == sd].to_list()[0]
                end_idx = date.index[date['date'] == ed].to_list()[-1]
                
                y = Data[y_name].iloc[start_idx:end_idx]
                X = date['date'].iloc[start_idx:end_idx]

                with col12:
                    ws = st.slider('Smoothness', 4.0, 27.0, 4.0, step=1.0, key=f'slider{i}')
                    if smooth:
                        y = savgol_filter(y, int(ws), 3)

                fig = px.line(x=X, y=y,  
                                    range_y=[ymin, ymax], 
                                    )
                st.plotly_chart(fig, use_container_width=True, key=f'time_plot{i}')


    with tab4:
        st.subheader('Bar Charts', divider=True)
        c1, c2 = st.columns([1, 4])
        with c1:
            col_number = st.number_input('number of columns to compare ...', value=24, key="col_number")

        d = pd.concat([y, D['sensor_05'], D['sensor_36']], axis=1).iloc[:col_number]
        sc = StandardScaler()
        d = pd.DataFrame(sc.fit_transform(d), columns=['sensor_28', 'snesor_05', 'sensor_36'])

        col1, col2, col3, col4 = st.columns([1,1,1,12])

        with col1:
            color1 = st.color_picker("sensor_28", "#FF3639", key='bar_color1')
        with col2:
            color2 = st.color_picker("sensor_05", "#70E9F9", key='bar_color2')
        with col3:
            color3 = st.color_picker("sensor_36", "#179BDE", key='bar_color3')
        st.bar_chart(d, color=[color1, color2, color3])

        d2 = Data[['sensor_28', 'sensor_05']].iloc[:col_number]
        d2 = pd.concat([d2, D['sensor_36']], axis=1).iloc[:col_number]
        sc2 = MinMaxScaler()
        d2 = pd.DataFrame(sc2.fit_transform(d2), columns=['sensor_28', 'snesor_05', 'sensor_36'])

        col11, col22, col33, col44 = st.columns([1,1,1,12])

        with col11:
            color11 = st.color_picker("sensor_28", "#FF8336", key='bar_color11')
        with col22:
            color22 = st.color_picker("sensor_05", "#70E9F9", key='bar_color22')
        with col33:
            color33 = st.color_picker("sensor_36", "#179BDE", key='bar_color33')
        st.bar_chart(d2, color=[color11, color22, color33])

