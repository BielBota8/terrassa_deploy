import pandas as pd
import os
from SIAD_preprocessing import siad_preprocess
from MENJADOR_preprocessing import menja_preprocessing
from JOIN_preprocessing import join_preprocessing    

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



def write_data_to_folder(dades_discriminacions, dades_ajuts_menjador):
    """
    Writes the discrimination and meal assistance data to CSV files in the user's Downloads folder.

    Parameters:
    dades_discriminacions (pandas.DataFrame): DataFrame containing the discrimination data.
    dades_ajuts_menjador (pandas.DataFrame): DataFrame containing the meal assistance data.
    """
    data_path = os.path.join(os.path.expanduser('~'), 'Downloads')

    if dades_discriminacions is not None:
        dades_discriminacions.to_csv(os.path.join(data_path, 'dades_discriminacions.csv'), index=False)


    if dades_ajuts_menjador is not None:
        dades_ajuts_menjador.to_csv(os.path.join(data_path, 'dades_ajuts_menjador.csv'), index=False)

    if dades_discriminacions is not None and dades_ajuts_menjador is not None:
        merged = join_preprocessing(dades_discriminacions, dades_ajuts_menjador)
        merged.to_csv(os.path.join(data_path, 'dades_merged.csv'), index=False)

