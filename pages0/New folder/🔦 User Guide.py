import time
import streamlit as st

with st.spinner('Wait for it...'):
    time.sleep(5)
st.success("Done!")


import streamlit.components.v1 as components

components.html(
    "<p style='font-size: 50px;'><span style='text-decoration: line-through double red;'>Oops</span>!</p>"
)