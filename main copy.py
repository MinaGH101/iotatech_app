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
    tabs = on_hover_tabs(tabName=['home', 'IOTATECH', 'Data Trend View', 'Data Exploration', 'Prediction', 'Prediction R&D', 'Help'], 
                         iconName=['psychology', 'data_thresholding', 'settings_applications', 'factory', 'science', 'help'], default_choice=0)


# Function to dynamically import a module
def load_page(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if tabs == "home":
    if "home_module" not in st.session_state:
        st.session_state["home_module"] = load_page("home", "pages0/IOTATECH.py")
    # home_module = load_page("home", "pages0/IOTATECH.py")  # Load home.py


# if tabs=='IOTATECH':
#     tab1, tab2 = st.tabs(['Basic Information', 'Machine Learning Results'])

#     with tab1:


#         # st.page_link("pages0/Data Trend View.py", label='home', icon=":material/thumb_up:")

#         st.header("⚙️ Iota tech introduction")
#         # st.page_link("pages0/Data Trend View.py", label='home', icon=":material/thumb_up:")

#         col1, col2 = st.columns([1, 1])
#         with col1:
#             st.subheader('The Iota AI Group began its work in 2020 with the aim of developing and applying AI in industry. This group, composed of specialists in AI, materials and metallurgy engineers, chemical engineers, and software engineers, focuses on providing and implementing solutions to industrial challenges using AI tools.')
#         with col2:
#             image11 = Image.open('images/signature.png')
#             st.image(image11)

#         col11, col22 = st.columns([1, 1])
#         with col11:
#             image11 = Image.open('images/img80.jpg')
#             st.image(image11)
#         with col22:
#             st.subheader('The Iota AI Group leverages advanced AI tools to analyze and utilize industry-specific data, aiming to address the unique challenges and optimize processes within each sector. By tailoring AI-driven solutions to the distinct needs of various industries, the group strives to enhance efficiency, resolve complex issues, and drive innovation.')




#     with tab2:
#         st.subheader('correlation matrix plot:')
#         image1 = Image.open('images/Correlation2.jpg')
#         st.image(image1)
#         st.caption("this plot shows the pierson correlation between different attributes. higher correlation (+ or -), between two attributes means linear relationship.")
#         st.write('The surface area column exhibits the highest correlation with the columns representing the initial surface area and the sum of octahedral oxides. The correlation with the initial surface area might be attributed to an increase in the initial clay surface area, leading to enhanced contact between the acidic solution and the clay, resulting in increased ion exchange and better activation of the clay. Additionally, the correlation with the sum of octahedral oxides could be due to the increased presence of exchangeable ions. As the concentration of ions like Al3+ increases, the release of these ions from the octahedral structure of clay intensifies, leading to an increase in the final surface area. ')
        
#         container = st.container()
#         container.write('------------------------------------')
#         st.subheader('Model learning curve plot:')
#         image2 = Image.open('images/learning_curve.jpg')
#         st.image(image2)
#         st.caption("learning curve shows the model performence based on different amount of data.")
#         st.write('After data collection and preprocessing, various machine learning models were trained, and the gradient boosting model achieved the best result with an accuracy of 0.7 on the entire dataset. The above figure illustrates the learning curve of the model showing that, up to approximately 250 samples, the model performance is not satisfactory. However, as the number of data points increases, the accuracy of the model improves gradually, and its error converges.')

#         container = st.container()
#         container.write('------------------------------------')
#         st.subheader('y_true - y_pred plot:')
#         image3 = Image.open('images/pred.jpg')
#         st.image(image3, width=500)
#         st.caption("y_true - y_pred plot, shows that how good the trained model is, in prediction surface area of new data.")
#         st.write('In the above figure, it can be observed that the data points are relatively close to the green line, which has a slope of one. This indicates that the model has performed well in predicting the outcomes. Additionally, from the learning curve, it can be seen that as the number of data points increases, the model accuracy improves, and its error decreases. This confirms the model growth and improvement with an increase in the volume of data, highlighting the significance of expanding the dataset to enhance the model performance.')

#         container = st.container()
#         container.write('------------------------------------')
#         st.subheader('SHAP values plot:')
#         image4 = Image.open('images/SHAP_beeswarm2_test2.jpg')
#         st.image(image4)
#         st.caption("SHAP values plot, indicates the importance of each feature, based on the trained model result and rules of game theory")
#         st.write('The SHAP values (Shapley) represent the importance of each feature. It is observed that the acid concentration and the sum of octahedral oxides have the most significant impact on the surface area. Increasing their values has led to an increase in the final surface area.')



if tabs =='Data Trend View':
    st.header('📈 Data Trend View')
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
















elif tabs == 'Data Exploration':
    c1, c2, c3= st.columns([2, 1, 1])
    with c1:
        st.header('Data Exploration')
    with c2:
        scale = st.radio('Render', options=['high', 'medium', 'low'], horizontal=True, key='scale1')
    with c3:
        scale = st.radio('Digit accuracy', options=['1', '2', '3'], horizontal=True, key='scale2')

    # html_code = """
    # <div class="align-center justify-item-center">
    #     <h3>    <img style="width:50px;" src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KCjwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIEdlbmVyYXRvcjogU1ZHIFJlcG8gTWl4ZXIgVG9vbHMgLS0+Cjxzdmcgd2lkdGg9IjgwMHB4IiBoZWlnaHQ9IjgwMHB4IiB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxnIGlkPSJTeXN0ZW0gLyBEYXRhIj4KPHBhdGggaWQ9IlZlY3RvciIgZD0iTTE4IDEyVjE3QzE4IDE4LjY1NjkgMTUuMzEzNyAyMCAxMiAyMEM4LjY4NjI5IDIwIDYgMTguNjU2OSA2IDE3VjEyTTE4IDEyVjdNMTggMTJDMTggMTMuNjU2OSAxNS4zMTM3IDE1IDEyIDE1QzguNjg2MjkgMTUgNiAxMy42NTY5IDYgMTJNMTggN0MxOCA1LjM0MzE1IDE1LjMxMzcgNCAxMiA0QzguNjg2MjkgNCA2IDUuMzQzMTUgNiA3TTE4IDdDMTggOC42NTY4NSAxNS4zMTM3IDEwIDEyIDEwQzguNjg2MjkgMTAgNiA4LjY1Njg1IDYgN002IDEyVjciIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9nPgo8L3N2Zz4=" alt="Logo" width="100">
    # Data Exploration</h3>
    # </div>
    # """

    # # Render the HTML code
    # components.html(html_code, height=200)  

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


    y_ = st.selectbox('Target Column', options=Data.columns)
    y = Data[y_]

    import pickle as pk
    from sklearn.linear_model import ARDRegression
    model = ARDRegression(alpha_1=0.000001, alpha_2=0.000001, lambda_1=0.000001, lambda_2=0.000001)
    model.fit(X, y)

    # with open('model_md_5h.pkl', 'wb') as f:
    #     pk.dump(model, open(f'./model_md.pkl','wb'))


    tab1, tab2, tab3, tab4 = st.tabs(['Data Descriptipn', 'Linear Relations', 'Non-Linear Relations', 'Feature Importance'])

    with tab1:
        st.subheader('Data Description Table')
        desc= Data.describe()
        st.dataframe(desc)

        # full_report = st.checkbox('Get Full Report', value=False, key='full_report')
        # if full_report:
        #     options = st.multiselect(
        #     "Parameters to Report",
        #     list(Data.columns),
        #     list(Data.columns)[:7])
        #     rep = ProfileReport(Data[options])
        #     st_profile_report(rep)
            # st.write(rep.html, unsafe_allow_html = True)

    with tab2:

        st.subheader('Data Parameters Linear Relations')
        method = st.radio('Correlation Method', options=['pearson', 'kendall', 'spearman'], horizontal=True)
        corr = Data.corr(method = method)
        st.dataframe(corr)

        st.divider()

        options = st.multiselect(
        "Parameters",
        list(Data.columns),
        list(Data.columns)[:7])
        method2 = st.radio('Correlation Method', options=['pearson', 'kendall', 'spearman'], horizontal=True, key='method2')
        corr_p = Data[options].corr(method=method2)
        fig = px.imshow(corr_p, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)


        # sns.set(style='white')

        # def corrfunc(x, y, **kws):
        #     r, p = stats.pearsonr(x, y)
        #     p_stars = ''
        #     if p <= 0.05:
        #         p_stars = '*'
        #     if p <= 0.01:
        #         p_stars = '**'
        #     if p <= 0.001:
        #         p_stars = '***'
        #     ax = plt.gca()
        #     ax.annotate('r = {:.2f} '.format(r) + p_stars,
        #                 xy=(0.05, 0.9), xycoords=ax.transAxes)

        # def annotate_colname(x, **kws):
        #     ax = plt.gca()
        #     ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,
        #                 fontweight='bold')
        
        # def cor_matrix(df):
        #     g = sns.PairGrid(df, height=1.5)
        #     # Use normal regplot as `lowess=True` doesn't provide CIs.
        #     g.map_upper(sns.regplot, scatter_kws={'s':10})
        #     g.map_diag(sns.histplot, kde=True, kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4))
        #     g.map_diag(annotate_colname)
        #     g.map_lower(sns.kdeplot, cmap='Blues_d')
        #     g.map_lower(corrfunc)
        #     for ax in g.axes.flatten():
        #         ax.set_ylabel('')
        #         ax.set_xlabel('')

        #     return g

        # k = KNNImputer(n_neighbors=3)
        # Data_corr = Data[options]
        # fig = cor_matrix(pd.DataFrame(k.fit_transform(Data_corr), columns=options))
        # st.pyplot(fig)




    with tab3:
        st.subheader('Probability Relation')
        n_nei = st.number_input('n_neighbors', value=3, key='n_nei')
        mi = mutual_info_regression(X, y, n_neighbors=n_nei)
        mi = pd.Series(mi)
        mi.index = X.columns
        mi.sort_values(ascending=False).plot.bar(figsize=(50, 7))
        # plt.ylabel('Mutual Information')
        fig = px.bar(mi)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader('Polynomial Relation')
        univariate = f_regression(X.fillna(0), y)
        univariate = pd.Series(univariate[1])
        univariate.index = X.columns
        univariate.sort_values(ascending=True).plot.bar(figsize=(50,7))
        fig = px.bar(univariate)
        st.plotly_chart(fig, use_container_width=True)








    with tab4:
        st.header('Feature Importance - model based')
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
        mean_shap_values = shap_values_df.abs().mean().sort_values(ascending=False)


        n_f = st.number_input('n_features', value=10, key='n_f')
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        top_15_features = np.argsort(mean_abs_shap)[-1*n_f:]

        shap_values_top15 = shap_values.values[:, top_15_features]
        feature_names_top15 = X.columns[top_15_features]
        feature_values_top15 = X.values[:, top_15_features]

        fig2 = go.Figure()

        for i, feature_name in enumerate(feature_names_top15):
            fig2.add_trace(
                go.Scatter(
                    x=shap_values_top15[:, i],
                    y=[i] * shap_values_top15.shape[0],
                    mode='markers',
                    marker=dict(
                        size=7,
                        color=feature_values_top15[:, i],
                        colorscale='RdBu',
                        reversescale=True,
                        showscale=True if i == 0 else False,
                        colorbar=dict(title="Feature Value", x=1.1) if i == 0 else None,
                    ),
                    name=feature_name,
                )
            )

        fig2.update_layout(
            title="Top Important Features",
            xaxis_title="SHAP value",
            yaxis=dict(
                tickvals=list(range(len(feature_names_top15))),
                ticktext=feature_names_top15,
                title="Feature",
                showgrid=True,
                automargin=True,
            ),
            template="plotly_white",
            height=700,
            showlegend=False,
        )

        st.plotly_chart(fig2)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=mean_shap_values.values,
                    y=mean_shap_values.index,
                    orientation='h',
                    marker=dict(color='skyblue'),
                )
            ]
        )
        fig.update_layout(
            title="Mean SHAP Values",
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Feature",
            template="plotly_white",
        )

        st.plotly_chart(fig)
























elif tabs == 'Process Prediction':
    with open("images/hmi.svg", "r") as svg_file:
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
        st.session_state.M = pd.read_excel('./data/ais_metal_total_5.0_lag_ordered_new.xlsx')
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
    st.header('⚙️ Process output prediction')


    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(f"<div>{svg_content}</div>", unsafe_allow_html=True)

    with col2:

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


    ##############################################################################################

    col11, col22 = st.columns([1, 4])

    with col11:

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


if tabs=='Prediction R&D':
    st.header('📈 Process output prediction')

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


if tabs=='Prediction':

    with open("images/hmi.svg", "r") as svg_file:
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
        st.session_state.M = pd.read_excel('./data/ais_metal_total_5.0_lag_ordered_new.xlsx')
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
    st.header('⚙️ Process output prediction')


    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(f"<div>{svg_content}</div>", unsafe_allow_html=True)

    with col2:

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


    ##############################################################################################

    col11, col22 = st.columns([1, 4])

    with col11:

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

