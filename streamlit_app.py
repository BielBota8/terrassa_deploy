import streamlit as st
from datetime import datetime
import base64
from PIL import Image


from file_processor import read_dades_discriminacions
from file_processor import read_dades_ajuts_menjador
from file_processor import write_data_to_folder

from config import get_base64_image, create_stylish_sidebar, display_page_header

# Page configuration
st.set_page_config(
    page_title="Equivision Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo_base64 = get_base64_image("./assets/logo.png")
terrassa_base64=get_base64_image("./assets/terrassa.png")

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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Diccionari Dades</div>
                <p class="feature-description">
                    Diccionari que conté explicació de l'estructura de la base de dades.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">Manual</div>
                <p class="feature-description">
                    Manual per utilitzar correctament l'aplicació.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Decisions Pipeline</div>
                <p class="feature-description">
                    Explicació detallada del procés que seguim per netejar les dades.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
       
    # Getting started section
    st.markdown('<div class="sub-header">Com Començar</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="card">
            <ol>
                <li><strong>Carregar Dades:</strong> Utilitzi la secció "Carregar Dades" per pujar els seus arxius CSV o Excel.</li>
                <li><strong>Explorar Analítiques:</strong> Navegueu a "Visualització de Dades" per veure les visualitzacions i mètriques.</li>
                <li><strong>Personalitzar Visualitzacions:</strong> Filtra i ajusta els gràfics segons les teves necessitats.</li>
                <li><strong>Exportar Informes:</strong> Descarrega les visualitzacions o dades per compartir i prendre decisions.</li>
            </ol>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Footer
    st.markdown(
        """
        <div class="home-footer">
            <p>© 2025 Equivision. Tots els drets reservats.</p>
            <p>Versió 1.0.0 | Última actualització: Febrer 2025</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
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
    if st.button('Guardar Dades'):
        if 'dades_discriminacions' in st.session_state or 'dades_ajut_menjador' in st.session_state:
            # Pass the processed DataFrames from session_state, not the raw uploaded files
            write_data_to_folder(
                st.session_state.get('dades_discriminacions'),
                st.session_state.get('dades_ajut_menjador')
            )

            if 'dades_discriminacions' in st.session_state and st.session_state.dades_discriminacions is not None:
                st.success("Totes les dades han estat guardades correctament.")
            elif 'dades_ajut_menjador' in st.session_state and st.session:
                st.success("Dades d'ajuts menjador guardades correctament.")
            else:
                st.success("Dades de discriminacions guardades correctament.")
            
        else:
            st.error("No hi ha dades per guardar. Si us plau, carrega algun fitxer primer.")
    

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