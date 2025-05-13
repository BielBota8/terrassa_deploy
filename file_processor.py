import pandas as pd
import os
from SIAD_preprocessing import siad_preprocess
from MENJADOR_preprocessing import menja_preprocessing
from JOIN_preprocessing import join_preprocessing   
import streamlit as st
import io 

def read_dades_discriminacions(data):
    """
    Reads data from different files ("csv", "xls", "xlsx", "xlsb", "txt") containing discrimination records.

    Parameters:
        data (UploadedFile): Uploaded file object

    Returns:
        pandas.DataFrame: DataFrame containing the discrimination data.
    """

    data_df = None

    # Identificar el format del fitxer basant-se en el seu nom
    filename = data.name  # Agafar el nom del fitxer

    if filename.endswith('.csv'):
        # Try to peek at the first few bytes to check if the file starts with a semicolon
        first_bytes = data.read(10)
        data.seek(0)  # Reset file position
        
        # If file starts with semicolon, skip the first row
        if first_bytes.startswith(b';'):
            data_df = pd.read_csv(data, encoding='latin1', sep=';', skiprows=1)
        else:
            data_df = pd.read_csv(data, encoding='latin1', sep=';')
    elif filename.endswith('.xls') or filename.endswith('.xlsx'):
        data_df = pd.read_excel(data)
    elif filename.endswith('.xlsb'):
        data_df = pd.read_excel(data, engine='pyxlsb')
    elif filename.endswith('.txt'):
        data_df = pd.read_csv(data, sep='\t')

    return siad_preprocess(data_df)

            

def read_dades_ajuts_menjador(data):
    """
    Reads data from different files ("csv", "xls", "xlsx", "xlsb", "txt") containing aid records.

    Parameters:
        data (UploadedFile): Uploaded file object

    Returns:
        pandas.DataFrame: DataFrame containing the aid data.
    """

    data_df = None

    filename = data.name  # Agafar el nom del fitxer

    if filename.endswith('.csv'):
        data_df = pd.read_csv(data)
    elif filename.endswith('.xls') or filename.endswith('.xlsx'):
        data_df = pd.read_excel(data)
    elif filename.endswith('.xlsb'):
        data_df = pd.read_excel(data, engine='pyxlsb')
    elif filename.endswith('.txt'):
        data_df = pd.read_csv(data, sep='\t')

    return menja_preprocessing(data_df)



def show_download_buttons(dades_discriminacions, dades_ajuts_menjador):
    """
    Muestra botones de descarga en la interfaz de Streamlit para los datos proporcionados.
    """
    if dades_discriminacions is not None:
        csv_discriminacions = dades_discriminacions.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar dades_discriminacions.csv",
            data=csv_discriminacions,
            file_name="dades_discriminacions.csv",
            mime='text/csv'
        )

    if dades_ajuts_menjador is not None:
        csv_ajuts = dades_ajuts_menjador.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar dades_ajuts_menjador.csv",
            data=csv_ajuts,
            file_name="dades_ajuts_menjador.csv",
            mime='text/csv'
        )

    if dades_discriminacions is not None and dades_ajuts_menjador is not None:
        merged = join_preprocessing(dades_discriminacions, dades_ajuts_menjador)
        csv_merged = merged.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar dades_merged.csv",
            data=csv_merged,
            file_name="dades_merged.csv",
            mime='text/csv'
        )