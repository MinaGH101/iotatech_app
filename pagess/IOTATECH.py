def render_iota():

    import streamlit as st
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    from PIL import Image
    # import pages
    import pathlib
    from pathlib import Path
    from streamlit_option_menu import option_menu

    # with st.sidebar:
    #     st.markdown("[Home :material/thumb_up:](pages0/Data Trend View.py)")
    #     # st.page_link("pages0/Data Trend View.py", label='home', icon=":material/thumb_up:")



    # st.set_page_config(
    #     page_title="IOTA Tech Dashboard",
    #     page_icon="🏭",
    #     layout="wide",
    #     initial_sidebar_state="expanded",
    # )


    # pg = st.navigation([st.Page("IOTATECH.py"), st.Page("pages0/Prediction.py"), st.Page('pages0/Recommendation.py')])
    # pg.run()

    # icon_home = Path("./logo.png").as_posix()




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




    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")


    st.header('Iota AI Group')
    # st.title('IOTA Tech Dashboard')
    tab1, tab2 = st.tabs(['Basic Information', 'Machine Learning Results'])

    with tab1:


        # st.page_link("pages0/Data Trend View.py", label='home', icon=":material/thumb_up:")

        st.header("⚙️ Iota tech introduction")
        # st.page_link("pages0/Data Trend View.py", label='home', icon=":material/thumb_up:")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader('The Iota AI Group began its work in 2020 with the aim of developing and applying AI in industry. This group, composed of specialists in AI, materials and metallurgy engineers, chemical engineers, and software engineers, focuses on providing and implementing solutions to industrial challenges using AI tools.')
        with col2:
            image11 = Image.open('images/signature.png')
            st.image(image11)

        col11, col22 = st.columns([1, 1])
        with col11:
            image11 = Image.open('images/img80.jpg')
            st.image(image11)
        with col22:
            st.subheader('The Iota AI Group leverages advanced AI tools to analyze and utilize industry-specific data, aiming to address the unique challenges and optimize processes within each sector. By tailoring AI-driven solutions to the distinct needs of various industries, the group strives to enhance efficiency, resolve complex issues, and drive innovation.')




    with tab2:
        st.subheader('correlation matrix plot:')
        image1 = Image.open('images/Correlation2.jpg')
        st.image(image1)
        st.caption("this plot shows the pierson correlation between different attributes. higher correlation (+ or -), between two attributes means linear relationship.")
        st.write('The surface area column exhibits the highest correlation with the columns representing the initial surface area and the sum of octahedral oxides. The correlation with the initial surface area might be attributed to an increase in the initial clay surface area, leading to enhanced contact between the acidic solution and the clay, resulting in increased ion exchange and better activation of the clay. Additionally, the correlation with the sum of octahedral oxides could be due to the increased presence of exchangeable ions. As the concentration of ions like Al3+ increases, the release of these ions from the octahedral structure of clay intensifies, leading to an increase in the final surface area. ')
        
        container = st.container()
        container.write('------------------------------------')
        st.subheader('Model learning curve plot:')
        image2 = Image.open('images/learning_curve.jpg')
        st.image(image2)
        st.caption("learning curve shows the model performence based on different amount of data.")
        st.write('After data collection and preprocessing, various machine learning models were trained, and the gradient boosting model achieved the best result with an accuracy of 0.7 on the entire dataset. The above figure illustrates the learning curve of the model showing that, up to approximately 250 samples, the model performance is not satisfactory. However, as the number of data points increases, the accuracy of the model improves gradually, and its error converges.')

        container = st.container()
        container.write('------------------------------------')
        st.subheader('y_true - y_pred plot:')
        image3 = Image.open('images/pred.jpg')
        st.image(image3, width=500)
        st.caption("y_true - y_pred plot, shows that how good the trained model is, in prediction surface area of new data.")
        st.write('In the above figure, it can be observed that the data points are relatively close to the green line, which has a slope of one. This indicates that the model has performed well in predicting the outcomes. Additionally, from the learning curve, it can be seen that as the number of data points increases, the model accuracy improves, and its error decreases. This confirms the model growth and improvement with an increase in the volume of data, highlighting the significance of expanding the dataset to enhance the model performance.')

        container = st.container()
        container.write('------------------------------------')
        st.subheader('SHAP values plot:')
        image4 = Image.open('images/SHAP_beeswarm2_test2.jpg')
        st.image(image4)
        st.caption("SHAP values plot, indicates the importance of each feature, based on the trained model result and rules of game theory")
        st.write('The SHAP values (Shapley) represent the importance of each feature. It is observed that the acid concentration and the sum of octahedral oxides have the most significant impact on the surface area. Increasing their values has led to an increase in the final surface area.')

