import streamlit as st

create_page = st.Page("ddd.py", title="Create entry", icon=":material/add_circle:")
delete_page = st.Page("sss.py", title="Delete entry", icon=":material/delete:")

pg = st.navigation([create_page, delete_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style2.css")
