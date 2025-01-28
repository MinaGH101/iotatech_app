import streamlit as st 
import numpy as np 
import pandas as pd
from tools import *
# from ydata_profiling import ProfileReport
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
import plotly.express as px
# from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import shap
import joblib
import plotly.graph_objects as go


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

st.header('🔎 Data Exploration')
if 'D' not in st.session_state:
    st.session_state.D = pd.read_csv('./pump_data.csv')
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
