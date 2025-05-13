import streamlit as st
from datetime import datetime
import base64
from PIL import Image

from file_processor import read_dades_discriminacions
from file_processor import read_dades_ajuts_menjador
from JOIN_preprocessing import join_preprocessing

from config import get_base64_image, create_stylish_sidebar, display_page_header

# Page configuration
st.set_page_config(
    page_title="Equivision Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo_base64 = get_base64_image("./assets/logo.png")
terrassa_base64 = get_base64_image("./assets/terrassa.png")

# Create a menu with three options
menu = create_stylish_sidebar(logo_base64, terrassa_base64)

# Initialize session state for uploaded data
if 'dades_discriminacions' not in st.session_state:
    st.session_state.dades_discriminacions = None
if 'dades_ajut_menjador' not in st.session_state:
    st.session_state.dades_ajut_menjador = None
if 'use_sample_data' not in st.session_state:
    st.session_state.use_sample_data = True

# Main content area based on menu selection
if menu == "Inici":
    # Home page content
    display_page_header(terrassa_base64)
    st.markdown(
        """
        <div class="home-hero">
            <h1>Benvinguts a Equivision</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    # Home page features and footer (omitted for brevity)

elif menu == "Carregar Dades":
    display_page_header(terrassa_base64)
    st.markdown('<div class="main-header">Carregar Dades</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="sub-header">Pujar Fitxers de Dades</div>', unsafe_allow_html=True)
    
    # File upload section
    dades_discriminacinos = st.file_uploader("Pujar Dades de Discriminacions", type=["csv", "xls", "xlsx", "xlsb", "txt"])
    
    if dades_discriminacinos is not None:
        try:
            st.session_state.dades_discriminacions, missing = read_dades_discriminacions(dades_discriminacinos)
            if len(missing) == 0:
                st.success(f"Dades carregades correctament: {dades_discriminacinos.name}")
            else:
                st.warning(f"Falten columnes {missing}, les dades preprocessades pot no ser correctes")
            st.dataframe(st.session_state.dades_discriminacions.head())
        except Exception as e:
            st.error(f"Error en carregar les dades: {e}")

    # Customer data upload
    dades_ajuts_menjador = st.file_uploader("Pujar Dades d'Ajuts Menjador Excel", type=["csv", "xls", "xlsx", "xlsb", "txt"])

    if dades_ajuts_menjador is not None:
        try:
            # Process the uploaded file to create a DataFrame
            st.session_state.dades_ajut_menjador = read_dades_ajuts_menjador(dades_ajuts_menjador)
            st.success(f"Dades carregades correctament: {dades_ajuts_menjador.name}")
            st.dataframe(st.session_state.dades_ajut_menjador.head())
        except Exception as e:
            st.error(f"Error en carregar les dades: {e}")

    # Save data button with verification check
    st.markdown("### Guardar Dades")

    dades_discriminacions = st.session_state.get('dades_discriminacions')
    dades_ajut_menjador = st.session_state.get('dades_ajut_menjador')

    if dades_discriminacions is not None:
        try:
            csv_discriminacions = dades_discriminacions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descarregar dades_discriminacions.csv",
                data=csv_discriminacions,
                file_name="dades_discriminacions.csv",
                mime='text/csv',
                key="download_discriminacions"
            )
        except Exception as e:
            st.error(f"Error al generar el CSV de dades_discriminacions: {e}")

    if dades_ajut_menjador is not None:
        try:
            csv_ajuts = dades_ajut_menjador.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descarregar dades_ajuts_menjador.csv",
                data=csv_ajuts,
                file_name="dades_ajuts_menjador.csv",
                mime='text/csv',
                key="download_ajuts"
            )
        except Exception as e:
            st.error(f"Error al generar el CSV de dades_ajuts_menjador: {e}")

    if dades_discriminacions is not None and dades_ajut_menjador is not None:
        try:
            merged = join_preprocessing(dades_discriminacions, dades_ajut_menjador)
            csv_merged = merged.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descarregar dades_merged.csv",
                data=csv_merged,
                file_name="dades_merged.csv",
                mime='text/csv',
                key="download_merged"
            )
        except Exception as e:
            st.error(f"Error al generar el CSV de dades_merged: {e}")


    st.markdown("</div>", unsafe_allow_html=True)
    
    # Data selection explanation
    st.info(
        "Les dades carregades s'utilitzaran a la secció 'Visualització de Dades'. "
        "Si no es carreguen dades, s'utilitzaran les dades de mostra per a la visualització."
    )
    
elif menu == "Visualització de Dades":
    display_page_header(terrassa_base64)
    st.markdown('<div class="main-header">Visualització de Dades</div>', unsafe_allow_html=True)
    
    # Initialize session state variables if they don't exist
    if 'powerbi_url' not in st.session_state:
        st.session_state.powerbi_url = ""
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False
    
    # Create a container with centered content
    st.markdown("""
        <style>
        .centered-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Use columns with empty spaces on the sides to center the content
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            powerbi_url = st.text_input("URL de l'informe Power BI", 
                                        value=st.session_state.powerbi_url,
                                        placeholder="Introduïu l'URL de l'informe Power BI",
                                        key="url_input")
            st.session_state.powerbi_url = powerbi_url
            
        with col2:
            # Add some vertical space to align with the input field
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Visualitzar informe"):
                st.session_state.show_report = True
    
    # Display the Power BI report if button is clicked and URL is provided
    if st.session_state.show_report and st.session_state.powerbi_url:
        try:
            st.components.v1.iframe(st.session_state.powerbi_url, width=1000, height=800)
        except Exception as e:
            st.error(f"Error en carregar l'informe: {e}")
            st.info("Si us plau, verifiqueu que l'URL sigui correcte i que tingui el format adequat per a Power BI embed.")
    elif st.session_state.show_report and not st.session_state.powerbi_url:
        st.warning("Si us plau, introduïu un URL d'informe Power BI vàlid.")