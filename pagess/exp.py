from tools import *
def render_exp():
    import streamlit as st 
    import numpy as np 
    import pandas as pd
    
    # from ydata_profiling import ProfileReport
    from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
    import plotly.express as px
    # from streamlit_pandas_profiling import st_profile_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import shap
    import os
    import joblib
    import plotly.graph_objects as go
    import streamlit.components.v1 as components
    from streamlit_option_menu import option_menu

    st.cache_data.clear()

    # file = st.file_uploader("file")

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style.css")


    st.sidebar.markdown(
        f"""
        <style>
        .text-muted {{
        display: none;
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
        # from streamlit_shap import st_shap
        # import shap

        # from sklearn.model_selection import train_test_split
        # import xgboost

        # def load_data():
        #     return shap.datasets.adult()

        # def load_model(X, y):
        #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
        #     d_train = xgboost.DMatrix(X_train, label=y_train)
        #     d_test = xgboost.DMatrix(X_test, label=y_test)
        #     params = {
        #         "eta": 0.01,
        #         "objective": "binary:logistic",
        #         "subsample": 0.5,
        #         "base_score": np.mean(y_train),
        #         "eval_metric": "logloss",
        #         "n_jobs": -1,
        #     }
        #     model = xgboost.train(params, d_train, 10, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
        #     return model

        # # train XGBoost model
        # X2,y2 = load_data()
        # X_display,y_display = shap.datasets.adult(display=True)

        # model2 = load_model(X2, y2)

        # # compute SHAP values
        # explainer = shap.Explainer(model2, X2)
        # shap_values = explainer(X2)

        # # st_shap(shap.plots.waterfall(shap_values[0]), height=300)
        # # st_shap(shap.plots.beeswarm(shap_values), height=300)

        # explainer = shap.TreeExplainer(model2)
        # shap_values = explainer.shap_values(X2)

        # st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_display.iloc[0,:]), height=200, width=1000)
        # st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_display.iloc[:1000,:]), height=400, width=1000)



#######################################################################
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
