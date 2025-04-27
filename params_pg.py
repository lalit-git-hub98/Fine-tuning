import streamlit as st

def navigate_to(page_name):
    st.session_state.page = page_name

def params_pg(model_name):
    st.title("Params Page")
    st.write(model_name)
    
