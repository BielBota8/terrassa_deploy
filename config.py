from templates.css import custom_css_sidebar, general_css
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def create_stylish_sidebar(logo_base64, terrassa_base64):
    general_css()
    custom_css_sidebar()
    

    st.sidebar.markdown(
        f"""
        <div style="text-align: center; padding: 1rem 0;">
            <img src="data:image/png;base64,{logo_base64}" width="120" style="margin-bottom: 10px;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; padding-bottom: 1rem;">
            <img src="data:image/png;base64,{terrassa_base64}" width="240" style="margin-bottom: 10px;">
            <p style="opacity: 0.7;">Analytics Dashboard</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    with st.sidebar:
        
        
        
        # Use option_menu for better looking navigation
        selected = option_menu(
            menu_title=None,  # No title needed
            options=["Inici", "Carregar Dades", "Visualització de Dades", "Sobre Preprocessat"],
            icons=["house", "cloud-upload", "bar-chart", "gear"],  # Bootstrap icons
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f5f7f9"},
                "icon": {"color": "#bd85f2", "font-size": "16px"}, 
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eaeaea"},
                "nav-link-selected": {"background-color": "#9a44eb"},
            }
        )
        
        # Add a footer section
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: #888; padding: 10px;'>Terrassa </div>", unsafe_allow_html=True)
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Data:** {current_time}")

    
    return selected

def display_page_header(terrassa_base64):
    col1, col2 = st.columns([3, 1])
    terrassa_base64=get_base64_image("./assets/terrassa.png")
    with col2:
        st.markdown(
            f"""
            <div style="text-align: left; padding-top: 10px;">
                <img src="data:image/png;base64,{terrassa_base64}" width="400">
            </div>
            """, 
            unsafe_allow_html=True
        )
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)