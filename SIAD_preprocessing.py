import pandas as pd
import numpy as np
import random
import re

def siad_preprocess(df):
    """
    Reads dataframe from SIAD and preprocesses it.

    Parameters:
        data (pd.DataFrame): Dataframe containing the SIAD data.

    Returns:
        pandas.DataFrame: Preprocessed discrimination data.
    """

    #Define cols to drop
    # cols_to_drop = ["COD_AREA", "DATA_BAIXA", "ESTAT",
    #             "NOM_ESTAT", "RESULTAT", "TIPUSCITA", "DENUNCIA",
    #             "EXP_ORIGEN", "COD_EXP_ECIVIL", "COD_EXP_NIVSOCIO",
    #             "COD_EXP_PROFESIO", "COD_EXP_TITULACIO", "FONT_INGRESSOS", "COD_IDGENERE",
    #             "COD_MOTIU", "COD_MUNICIPI", "COD_PERMIS_RESI", "COD_PERMIS_TREBALL", "COD_PER_ECIVIL",
    #             "COD_PER_NACIONALITAT", "COD_PER_NIVSOCIO", "COD_PER_PROFESIO",
    #             "COD_PER_TITULACIO", "COD_USUARI_RESPON", "BAJA_PH",
    #             "DATA_BAIXA_MAL", "COD_AREA_1", "DATA_BAIXA_2", "HORARI",
    #             "DESC_HORARI", "MOTIU_TANCAMENT", "D_OCUPACIO", "D_ORIGEN", "TIPUS_ALTA", "SEXE_PH"]
    
    # redundant_columns = ["DESC_AREA", "PA_COD_PERSONA", "COD_PERSONA_1", "COD_SUBEXPEDIENT_1",
    #                  "COD_PROVINCIA", "N_FILLS_PG"]
    
    # cols_to_drop_ph = ["DATA_NAIX_PH", "TITULACIO_PH", "PAIS_PROC_PH", "NACIONALITAT_PH"]
    
    # cols_to_drop_pg = ["POBLACION_PG", "D_ESTAT_CIVIL_PG", "D_PROFESION_PG", "D_NIV_SOCI_ECO_PG", "PROVINCIA_PG",
    #                "NACIONALITAT_PG", "PAIS_PROC_PG", "D_TITULACIO_PG", "B_RECIBIRINFORMACION_PG", "TIPUS_DOCUMENT_PG"]

    # cols_to_drop_per = ["NOM_PER_PROFESIO", "NOM_PER_NIVSOCIO", "NOM_PER_TITULACIO", "PER_NUMFILLS",
    #                 "NOM_PER_ECIVIL", "NOM_PER_NACIONALITAT"]

    # cols_with_no_info = ["PRIMERA_VISITA", "DATA_CITA_FI", "DATA_MODI", "COD_SUBEXPEDIENT",
    #                  "DATA_MODIFICACIO", "DATA_SOLICITUD",
    #                  "DATA_TANCAMENT", "DATA_ALTA_2", "DATA_MODI_2",
    #                  "DATA_TANCAMENT_1", "COD_GRUP", "NOM_RESULTAT",
    #                  "DATA_MODI_MAL", "COD_EXPEDIENT_1", "COD_PERSONA_2", "DATA_ALTA",
    #                  "ASSISTEIX_CURS", "CONJUNTA", "DESPLAZADA", "NOM_IDGENERE", "NOM_MUNICIPI",
    #                  "SEXE_PG", "B_TRADUCTOR_PG", "INFANCIA", "MALTRACTAMENT",
    #                  "CODI_PER_MAL", "AGRESSOR", "DATA_ALTA_MAL", "COD_PERSONA", "DATA_NAIX_PG",
    #                  "COD_EXP_MAL", "DATA_NAIX"]
    cols_important = ['ID_UPC', 'COD_CITA', 'DISTRICTE_PH', 'SECCIO_PH', 'BARRI_PH',
                'NOM_BARRI_PH', 'NOM_AREA', 'COD_EXPEDIENT', 'DATA_CITA_INI',
                'NOM_TIPUSCITA', 'DATA_CITA_FI', 'DATA_OBERTURA', 'DESC_DENUNCIA', 'EDAT',
                'EXP_NOM_ORIGEN', 'EXP_NUMFILLS', 'GR_EDAT', 'NOM_EXP_ECIVIL',
                'NOM_EXP_NIVSOCIO', 'NOM_EXP_PROFESIO', 'NOM_EXP_TITULACIO',
                'NOM_FONT_INGRESSOS', 'NOM_MOTIU', 'NOM_PERMIS_RESI', "NOM_PER_TITULACIO",
                'NOM_PERMIS_TREBALL', 'NOM_PER_PAISNAIX', 'TIPUSFAMILIA',
                'GRUPNACIONALITAT', 'COD_MAL', 'DESC_MOTIU_TANCAMENT',
                'DESC_TIPUS_ALTA', 'DISCAPACITAT', 'ORDRE PROTECCIO',
                'DESC_MALTRACTAMENT', 'DESC_AGRESSOR']
    
    missing = []
    try:
        df = df[cols_important]
    except KeyError as e:
        missing = re.findall(r"'(.*?)'", str(e))
        for missing_column in missing:
            df[missing_column] = np.nan
        df = df[cols_important]

    #Create column DURADA_CITA
    df["DATA_CITA_INI"] = pd.to_datetime(df["DATA_CITA_INI"], format='mixed')
    df["DATA_CITA_FI"] = pd.to_datetime(df["DATA_CITA_FI"], format='mixed')
    df["DATA_OBERTURA"] = pd.to_datetime(df["DATA_OBERTURA"], format='mixed')

    df["DURADA_CITA"] = df.apply(lambda col: col["DATA_CITA_FI"]-col["DATA_CITA_INI"]
                             if col["DATA_CITA_FI"] > col["DATA_CITA_INI"]
                             else col["DATA_CITA_INI"]-col["DATA_CITA_FI"], axis=1)
    if df["DURADA_CITA"].notna().all():
      df["DURADA_CITA"] = df['DURADA_CITA'].dt.total_seconds()/60

    df = df.drop(columns = ["DATA_CITA_FI"])

    #Get only the rows >2015
    df = df[(df["DATA_OBERTURA"] >= '01-01-2015')|df["DATA_OBERTURA"].isna()]

    #Drop columns defined previously
    df["NOM_EXP_TITULACIO"] = df["NOM_EXP_TITULACIO"].fillna(df["NOM_PER_TITULACIO"])
    df = df.drop(columns=["NOM_PER_TITULACIO"])

    # df = df.drop(columns=cols_to_drop+
    #                  redundant_columns+
    #                  cols_to_drop_ph+
    #                  cols_to_drop_pg+
    #                  cols_to_drop_per+
    #                  cols_with_no_info)

    #Synthetic values to new columns
    np.random.seed(1)
    if df["DISCAPACITAT"].isna().all():
        df["DISCAPACITAT"] = np.random.choice(['No', 'Si'], size=len(df), p=[0.98, 0.02])

    if df["ORDRE PROTECCIO"].isna().all():
        df["ORDRE PROTECCIO"] = df.apply(lambda row: 'No' if pd.isna(row['COD_MAL']) else 
                                         np.random.choice(['No', 'Si'], p=[0.86, 0.14]), axis=1)

    if df["DESC_MALTRACTAMENT"].isna().all():
        df["DESC_MALTRACTAMENT"] = df.apply(lambda row: 'No aplica' if pd.isna(row['COD_MAL']) else
                                 np.random.choice(["Maltractament fisic", "Maltractament global",
                                                   "Maltractament psicologic", "Maltracte sexual",
                                                   "Violència vicaria", "Mutilació genial femenina",
                                                   "Violència obstètrica", "Violència econòmica",
                                                   "Negligència", "Orientació Sexual"],
                                                    p=[0.2441,0.1034,0.5678,0.0508,0.0237,0.0051,0.0007,0.0034, 0.0005, 0.0005]), axis=1)

    if df["DESC_AGRESSOR"].isna().all():
        df["DESC_AGRESSOR"] = df.apply(lambda row: 'Sense dades' if pd.isna(row['COD_MAL']) else 
                                         np.random.choice(["Parella", "Ex-parella", "Pares", "Pare", "Àmbit sanitari",
                                                           "Familiar", "Fill/a", "Feina", "Conegut", "Desconegut"],
                                                            p=[0.6154, 0.2271, 0.0391, 0.0017, 0.0017, 0.0271, 0.0323, 0.0169, 0.0169, 0.0218]), axis=1)

    #Change NAs of categoricals to Sense dades
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna('Sense dades')

    #Fix wrong EDAT values
    ##Create function that gets the min and max from GR_EDAT
    def extract_min_max(range_str):
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            return start, end
        elif '>' in range_str:
            limit = int(range_str.strip('>'))
            return limit, 123
        else:
            return np.nan, np.nan 
        
    df[['MIN_EDAT', 'MAX_EDAT']] = df['GR_EDAT'].apply(
    lambda x: pd.Series(extract_min_max(x)))

    ##Fix EDAT based on min and max extracted
    df["EDAT"] = df.apply(
    lambda row: row['EDAT']
    if pd.isna(row['MIN_EDAT']) or pd.isna(row['MAX_EDAT']) or row['MIN_EDAT'] <= row['EDAT'] <= row['MAX_EDAT']
    else random.randint(row['MIN_EDAT'], row['MAX_EDAT']),
    axis=1
    )
    
    ##Drop columns min and max
    df = df.drop(columns = ["MIN_EDAT", "MAX_EDAT"])
            
    #Replace non-informative values to Sense dades
    df["DESC_DENUNCIA"] = df["DESC_DENUNCIA"].replace('NS', 'Sense dades')
    df["NOM_EXP_ECIVIL"] = df["NOM_EXP_ECIVIL"].replace('Sense informacio', 'Sense dades')
    df["NOM_EXP_PROFESIO"] = df["NOM_EXP_PROFESIO"].replace('Sense informacio', 'Sense dades')
    df["NOM_PERMIS_RESI"] = df["NOM_PERMIS_RESI"].replace('NS', 'Sense dades')
    df["NOM_PERMIS_TREBALL"] = df["NOM_PERMIS_TREBALL"].replace('Sense informacio', 'Sense dades')
    df["DESC_TIPUS_ALTA"] = df["DESC_TIPUS_ALTA"].replace('Sense informacio', 'Sense dades')
    df["GR_EDAT"] = df["GR_EDAT"].replace('>80', '80+')
        
    #Drop rows in which we have no information about DATA_OBERTURA
    df = df[(df["DATA_OBERTURA"] != '1900-01-01')]
    df = df.dropna(subset='DATA_OBERTURA')
    
    return (df, missing)
