
import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import KNNImputer
from PIL import Image
import plotly.figure_factory as ff
import pywt
from scipy.signal import savgol_filter
import plotly.express as px
from pygwalker.api.streamlit import StreamlitRenderer
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from st_aggrid import AgGrid
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
st.session_state.Data = Data
Date_time = st.session_state.date_time
date = pd.DataFrame()
date['date'] = pd.to_datetime(Date_time).dt.date
sd1 = date['date'].iloc[0]
ed1 = date['date'].iloc[-1]

tab00, tab0, tab1, tab2, tab3, tab4 = st.tabs(['Overview', 'Processing', '2D Plot', '3D Plot', 'Time Plot', 'Histogram'])



with tab00: 
    # df = pd.read_excel(Data) 
    p = StreamlitRenderer(Data)
    p.explorer()

    filt = st.checkbox('Filter data', key='filt')
    if filt:
        AgGrid(Data)

    options = st.multiselect(
    "Parameters",
    list(Data.columns),
    list(Data.columns)[:6])

    c1, c2, c3 = st.columns([1, 1, 1])

    for i in range(len(options)):
        if i%3 == 1:
            with c1:
                bin_size = st.number_input('bin size', key=f'bin{i}')
                c = options[i]
                fig = ff.create_distplot([Data[c].dropna().tolist()], group_labels=[c], bin_size=bin_size, show_hist=True, show_rug=False, 
                                         colors=["#098fb8"])
                fig.update_layout(
                    title_text=f"{c}",
                    # title_x=0.5,
                    xaxis_title="Value",
                    yaxis_title="Density",
                    font=dict(size=12),
                    )
                st.plotly_chart(fig)

        if i%3 == 2:
            with c2:
                bin_size = st.number_input('bin size', key=f'bin{i}')
                c = options[i]
                fig = ff.create_distplot([Data[c].dropna().tolist()], group_labels=[c], bin_size=bin_size, show_hist=True, show_rug=False, 
                                         colors=["#098fb8"])
                fig.update_layout(
                    title_text=f"{c}",
                    # title_x=0.5,
                    xaxis_title="Value",
                    yaxis_title="Density",
                    font=dict(size=12),
                    )
                st.plotly_chart(fig)

        if i%3 == 0:
            with c3:
                bin_size = st.number_input('bin size', key=f'bin{i}')
                c = options[i]
                fig = ff.create_distplot([Data[c].dropna().tolist()], group_labels=[c], bin_size=bin_size, show_hist=True, show_rug=False, 
                                         colors=["#098fb8"])
                fig.update_layout(
                    title_text=f"{c}",
                    # title_x=0.5,
                    xaxis_title="Value",
                    yaxis_title="Density",
                    font=dict(size=12),
                    )
                st.plotly_chart(fig)


with tab0:
    st.subheader('Data imputation')

    c1, c2 = st.columns([1, 5])
    with c1:
        st.write("Missing Values Overview")
        missing_counts = st.session_state.Data.isnull().sum()
        st.write(missing_counts)

    with c2:
        selected_columns = st.multiselect("Select columns to impute:", st.session_state.Data.columns, default=st.session_state.Data.columns)

        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            method = st.radio("Choose an imputation method:", 
                    ["KNN Imputer", "Interpolate", "Replace (0, Mean, Mode)", "Delete Rows"], index=0)
            
        with c2:
            if method == "KNN Imputer":
                neighbors = st.slider("Select number of neighbors (K):", 3, 15, 1)
                miss_val = st.number_input('Missing value', min_value=np.nan, placeholder='default: nan')

                if st.button("Apply KNN Imputer"):
                    imputer = KNNImputer(n_neighbors=neighbors, missing_values=miss_val)
                    st.session_state.Data[selected_columns] = imputer.fit_transform(st.session_state.Data[selected_columns])
                    st.success("KNN imputation applied successfully!")

            elif method == "Interpolate":
                interpolate_method = st.selectbox('interpolate method', 
                                                  options=['linear', 'spline', 'slinear', 'nearest','pad'],
                                                  key='interpolate_method')
                limit_direction = st.pills('direction limit', options=['forward', 'backward', 'both'], label_visibility='collapsed', key='limit_direction')
                limit_area = st.pills('area limit', options=['inside', 'outside'], label_visibility='collapsed', key='limit_area')

                if st.button("Apply Interpolation"):
                    st.session_state.Data[selected_columns] = st.session_state.Data[selected_columns].interpolate(method=interpolate_method,
                                                                                                                  limit_direction=limit_direction,
                                                                                                                  limit_area=limit_area)
                    st.success("Interpolation applied successfully!")

            elif method == "Replace (0, Mean, Mode)":
                replace_option = st.selectbox("Replace with:", ["0", "Mean", "Mode"])
                if st.button("Apply Replacement"):
                    for col in selected_columns:
                        if replace_option == "0":
                            st.session_state.Data[col].fillna(0, inplace=True)
                        elif replace_option == "Mean":
                            st.session_state.Data[col].fillna(st.session_state.Data[col].mean(), inplace=True)
                        elif replace_option == "Mode":
                            st.session_state.Data[col].fillna(st.session_state.Data[col].mode()[0], inplace=True)
                    st.success(f"Missing values replaced with {replace_option} successfully!")

            elif method == "Delete Rows":
                column_for_deletion = st.multiselect("Select columns to check for missing values:",
                                                     st.session_state.Data.columns,
                                                     default=st.session_state.Data.columns[:5])
                if st.button("Delete Rows"):
                    st.session_state.Data.dropna(subset=[column_for_deletion], inplace=True)
                    st.success(f"Rows with missing values in {column_for_deletion} deleted successfully!")

    pre1 = st.checkbox('preview imputed data', key='impute pre')
    if pre1:
        # st.subheader("Updated Data Preview")
        st.write(st.session_state.Data)

    ########################################################################
    st.subheader('Data Denoising')
    
    c11, c22 = st.columns([1, 0.1])
    with c11:

        if "denoising_sections" not in st.session_state:
            st.session_state.denoising_sections = []

        # Function to apply Savitzky-Golay filter
        def apply_savgol(data, window, poly, mode):
            return savgol_filter(data, window, poly, mode=mode)

        # Function to apply Wavelet denoising
        def apply_wavelet(data, wavelet, level):
            coeffs = pywt.wavedec(data, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Noise estimation
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
            coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            return pywt.waverec(coeffs_thresh, wavelet)[:len(data)]

        # Add new denoising section
        if st.button("➕ Add Denoising Section", key="add_section"):
            st.session_state.denoising_sections.append({
                "id": len(st.session_state.denoising_sections),
                "selected_columns": [],
                "method": "Savitzky-Golay",
                "window_size": 5,
                "mode": 'interp',
                "poly_order": 2,
                "wavelet": "db4",
                "wavelet_level": 1,
                "denoised_data": None  # Placeholder for previewing changes
            })

        # Display denoising sections
        for section in st.session_state.denoising_sections:
            section_id = section["id"]
            
            with st.expander(f"Denoising Section {section_id + 1}"):
                section["selected_columns"] = st.multiselect(
                    f"Select Parameters for Section {section_id + 1}",
                    st.session_state.Data.columns,
                    section["selected_columns"],
                    key=f"columns_{section_id}"
                )

                section["method"] = st.radio(
                    f"Choose Denoising Method for Section {section_id + 1}",
                    ["Savitzky-Golay", "Wavelet Denoising"],
                    index=0 if section["method"] == "Savitzky-Golay" else 1,
                    key=f"method_{section_id}"
                )

                if section["method"] == "Savitzky-Golay":
                    c0, c1, c2 = st.columns([1, 1, 1])
                    with c0:
                        section['mode'] = st.selectbox('filter mode', options=['mirror', 'constant', 'nearest', 'wrap', 'interp'], key=f"mode_{section_id}")
                    with c1:
                        section["window_size"] = st.slider(
                            f"Window Size (Section {section_id + 1})",
                            3, 31, section["window_size"], step=2, key=f"window_{section_id}"
                        )
                    with c2:
                        section["poly_order"] = st.slider(
                            f"Polynomial Order (Section {section_id + 1})",
                            1, 5, section["poly_order"], key=f"poly_{section_id}"
                        )
                else:
                    section["wavelet"] = st.selectbox(
                        f"Wavelet Type (Section {section_id + 1})",
                        pywt.wavelist(kind="discrete"),
                        index=pywt.wavelist(kind="discrete").index(section["wavelet"]),
                        key=f"wavelet_{section_id}"
                    )
                    section["wavelet_level"] = st.slider(
                        f"Wavelet Decomposition Level (Section {section_id + 1})",
                        1, 5, section["wavelet_level"],
                        key=f"level_{section_id}"
                    )

                # Apply denoising to preview changes (without modifying main Data)
                denoised_data = st.session_state.Data.copy()
                for col in section["selected_columns"]:
                    if section["method"] == "Savitzky-Golay":
                        denoised_data[col] = apply_savgol(
                            denoised_data[col], section["window_size"], section["poly_order"], section['mode']
                        )
                    else:
                        denoised_data[col] = apply_wavelet(
                            denoised_data[col], section["wavelet"], section["wavelet_level"]
                        )

                # Store preview data in session state
                section["denoised_data"] = denoised_data

                # Plot before and after denoising
                if section["selected_columns"]:
                    fig = px.line(denoised_data, y=section["selected_columns"], 
                                title=f"Denoised Parameters - Section {section_id + 1}")
                    st.plotly_chart(fig, use_container_width=True)

                # Save button - updates main Data
                if st.button(f"💾 Save Denoised Data (Section {section_id + 1})", key=f"save_{section_id}"):
                    for col in section["selected_columns"]:
                        st.session_state.Data[col] = denoised_data[col]
                    st.success(f"✅ Changes saved for {len(section['selected_columns'])} parameters!")

        pre2 = st.checkbox('preview denoised data', key='denoise pre')
        if pre2:
            # st.subheader("Updated Data Preview")
            st.write(st.session_state.Data)








####################################################################################
with tab1:
    c1, c2, c3, c4 = st.columns([1,1,1, 1])
    with c1:
        impute = st.selectbox('impute method', ['K_neighbors', 'remove' , 'mean', 'mode', 'interpolate'], key='impute1')

    with c2:
        if impute == 'K_neighbors':
            n_nei = st.slider('n neighbors', 0, 20, 5, step=1, key='knn1')

        if impute == 'interpolate':
            n_nei = st.selectbox('interpolate', ['linear', 'spline' , 'poly'], key='interpolate1')

    with c3:
        zeros = st.pills('how to handel zeros', ['replace with nan', 'stay still', 'mean'], key='zeros1')

    with c4:
        plot_number = st.number_input('number of plots to compare ...', value=1, key="plot_number1")

    for i in range(plot_number):
        st.subheader(f'Plot number {i+1}', divider=True)
        col0, col1, col2 = st.columns([6, 2 , 2])

        with col1:
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

        with col2:
            
            with st.form(f"my_form{i}", clear_on_submit=False):
                X_min = st.number_input("X_min", value=st.session_state.Data[X_name].min(), key=f"X_min{i}")
                X_max = st.number_input("X_max", value=st.session_state.Data[X_name].max(), key=f"X_max{i}")
                y_min = st.number_input("y_min", value=st.session_state.Data[y_name].min(), key=f"y_min{i}")
                y_max = st.number_input("y_max", value=st.session_state.Data[y_name].max(), key=f"y_max{i}")
                
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

        with col0:

            fig = px.scatter(st.session_state.Data, x=X_name, y=y_name, range_x=[Xmin, Xmax], range_y=[ymin, ymax], color=color)
            st.plotly_chart(fig, use_container_width=True, key=f'sc_plot{i}')



with tab2:

    c1, c2, c3, c4 = st.columns([1,1,1, 1])
    with c1:
        impute = st.selectbox('impute method', ['K_neighbors', 'remove' , 'mean', 'mode', 'interpolate'], key='impute2')

    with c2:
        if impute == 'K_neighbors':
            n_nei = st.slider('n neighbors', 0, 20, 5, step=1, key='knn2')

        if impute == 'interpolate':
            n_nei = st.selectbox('interpolate', ['linear', 'spline' , 'poly'], key='interpolate2')

    with c3:
        zeros = st.pills('how to handel zeros', ['replace with nan', 'stay still', 'mean'], key='zeros2')

    with c4:
        plot_number = st.number_input('number of plots to compare ...', value=1, key="plot_number2")



    for i in range(plot_number):
        
        st.subheader(f'Plot number {i+1}', divider=True)

        col0, col1, col2 = st.columns([5, 1.5, 1.5])

        with col1:
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
                
                X, y, z = st.session_state.Data[X_name], st.session_state.Data[y_name], st.session_state.Data[z_name]

                
        with col2:
            
            with st.form(f"my_form1{i}{i}", clear_on_submit=False):
                X_min = st.number_input("X_min", value=st.session_state.Data[X_name].min(), key=f"X_min2{i}{i}")
                X_max = st.number_input("X_max", value=st.session_state.Data[X_name].max(), key=f"X_max2{i}{i}")
                y_min = st.number_input("y_min", value=st.session_state.Data[y_name].min(), key=f"y_min2{i}{i}")
                y_max = st.number_input("y_max", value=st.session_state.Data[y_name].max(), key=f"y_max2{i}{i}")
                z_min = st.number_input("z_min", value=st.session_state.Data[z_name].min(), key=f"z_min2{i}{i}")
                z_max = st.number_input("z_max", value=st.session_state.Data[z_name].max(), key=f"z_max2{i}{i}")
                
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

        with col0:

            fig = px.scatter_3d(st.session_state.Data, x=X_name, y=y_name, z=z_name, 
                                range_x=[Xmin, Xmax], 
                                range_y=[ymin, ymax], 
                                range_z=[zmin, zmax], 
                                color=color)
            st.plotly_chart(fig, use_container_width=True, key=f'3d_plot{i}')



with tab3:

    c1, c2, c3, c4 = st.columns([1,1,1, 1])
    with c1:
        impute = st.selectbox('impute method', ['K_neighbors', 'remove' , 'mean', 'mode', 'interpolate'], key='impute3')

    with c2:
        if impute == 'K_neighbors':
            n_nei = st.slider('n neighbors', 0, 20, 5, step=1, key='knn3')

        if impute == 'interpolate':
            n_nei = st.selectbox('interpolate', ['linear', 'spline' , 'poly'], key='interpolate3')

    with c3:
        zeros = st.pills('how to handel zeros', ['replace with nan', 'stay still', 'mean'], key='zeros3')

    with c4:
        plot_number = st.number_input('number of plots to compare ...', value=1, key="plot_number3")



    for i in range(plot_number):

        st.subheader(f'Plot number {i+1}', divider=True)
        col0, col1 = st.columns([2, 6])

        with col0:

                y_name = st.selectbox(
                    'Select Y-axis',
                    (Data.columns), key=f"y_name3{i}{i}{i}")
                
                start_date = st.date_input("Start_Date", key=f'sd3{i}{i}{i}')
                end_date = st.date_input("End_Date", key=f'ed3{i}{i}{i}')


                with st.form(f"my_form3{i}{i}{i}", clear_on_submit=False):
                    y_min = st.number_input("y_min", value=st.session_state.Data[y_name].min(), key=f"y_min3{i}{i}{i}")
                    y_max = st.number_input("y_max", value=st.session_state.Data[y_name].max(), key=f"y_max3{i}{i}{i}")
                    

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
            
            df = st.session_state.Data.iloc[start_idx:end_idx]
            X = date['date'].iloc[start_idx:end_idx]

            with col12:
                ws = st.slider('Smoothness', 4.0, 27.0, 4.0, step=1.0, key=f'slider{i}')
                if smooth:
                    y = savgol_filter(y, int(ws), 3)

            fig = px.line(df, y=y_name,  
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

