import pandas as pd
import numpy as np

def menja_preprocessing(df):   
    df_filtered = df
    
    cols_to_drop = [
        "sollicitant_parentesc", "finques_urbanes_nr", "subcodi",  
        "codi_creuament", "discapacitat_fins_33",  
        "necessitats_geografiques", "redistribucio_equilibrada",  
        "municipi_id", "municipiescola_id",  
        "compactacio_ajut", "compactacio_dies", "compactacio_calendari_municipi",  
        "compactacio_completa", "complement_ajuntament", "complement_import_diari",  
        "no_enviar_creuament", "enviar_creuament", "no_processar_retorn",  
        "no_autoritzacio_creuament_discapacitat", "no_autoritzacio_creuament_familia",  
        "fora_termini_canvi_situacio", "recurs", "desestimat", "motiu_desestimat",  
        "requeriment", "requeriment_resposta", "manca_documentacio_requerint",  
        "deduccio_familia_monoparental", "ajut_lloguer",  
        "targeta_moneder", "percentatge_ajut_ajuntament", "nomes_ajut_municipal",  
        "dates_ajut_municipal", "escolaritzacio_compartida",  
        "autoritzacio_creuament_discapacitat", "autoritzacio_creuament_familia",  
        "import_justificat", "beca_garantida2", "subcodi", "sollicitant_concepte",  
        "tsi", "compactacio_data_inici", "compactacio_data_fi", "data_baixa",  
        "data_trasllat_comarca", "data_trasllat_altres_comarques",  
        "sobrevinguda_data", "requeriment_data_maxima", "beca_mecd_pagament",  
        "targeta_moneder_numero", "data_inici_ajut_municipal",  
        "data_fi_ajut_municipal", "pdfAltraDocumentacio1Form",  
        "pdfAltraDocumentacio2Form", "pdfAltraDocumentacio3Form",  
        "altes_multiples", "preu_real_concertada", "import_justificat_complement",
        "infants_acolliment_codi", 'sobrevinguda_valorada',
        'resultat_consulta_renda', 'resultat_consulta_renda0', 'resultat_consulta_renda1',
        'resultat_consulta_renda2', 'resultat_consulta_renda3', 'resultat_consulta_renda4',
        'resultat_consulta_renda5', 'resultat_consulta_renda6', 'resultat_consulta_renda7',
        'resultat_consulta_renda8', 'resultat_consulta_renda9', 'resultat_consulta_renda10',
        'resultat_consulta_renda11', 'resultat_consulta_renda12'
    ]
    
    df_filtered2 = df_filtered.drop(cols_to_drop, axis=1)
    
    columns_to_change = ['autoritzacio_aeat', 'familia_nombrosa_general', 'familia_nombrosa_especial', 'familia_monoparental', 'familia_monoparental_especial',
    'discapacitat_mes_33', 'habits_menjar_inadequats', 'trastorns_emocionals_familia', 'negligencia_lleu', 'seguiment_medic_excessiu', 'problemes_habitatge',
    'familia_monoparental_poc_suport', 'deduccions_geografiques', 'ajut_assignat', 'baixa_ajut', 'trasllat_comarca', 'trasllat_altres_comarques', 
    'nova_sollicitud_despres_assignacio', 'sobrevinguda', 'modificacio_dades_economiques', 'creuament_enviat', 'canvi_escola', 'educacio_especial_gratuitat',
    'compartida_nomes_1_progenitor', 'deduccio_infants_acolliment_familia_extensa_aliena', 'beca_mecd', 'alta_multiple', 'familia_sollicita_compactacio',
    'cap_membre_disposa_dni'
    ]
    
    for column in columns_to_change:
        df_filtered2[column] = df_filtered2[column].replace({0: 'No', 1: 'Si'})
    
    # columns_to_consider = [col for col in df_filtered2.columns if 'membre' not in col and 'memebre' not in col]
    # df_filtered2 = df_filtered2[columns_to_consider]
    
    df_filtered2['nivellescolar'] = df_filtered2['nivellescolar'].map({1: 'Infantil', 2: 'Primària', 3: 'ESO'})
    # df_filtered2['subvencio_lloguer_agencia_habitatge_catalunya'] = df_filtered2['subvencio_lloguer_agencia_habitatge_catalunya'].apply(lambda x: 1 if x > 0 else 0)
    # df_filtered2['ajut_urgencia_social'] = df_filtered2['ajut_urgencia_social'].apply(lambda x: 1 if x > 0 else 0)
    df_filtered2['risc_social'] = df_filtered2['risc_social'].map({0: 'Sense risc', 1: 'Risc social', 2: 'Risc social greu'})
    # df_filtered2['finques_rustiques'] = df_filtered2['finques_rustiques'].apply(lambda x: 1 if x > 0 else 0)
    # df_filtered2['finques_urbanes'] = df_filtered2['finques_urbanes'].apply(lambda x: 1 if x > 0 else 0)
    df_filtered2['renda_familiar'] = df_filtered2['renda_familiar'].apply(lambda x: -100000 if x < 0 else x)
    # df_filtered2['volum_negoci'] = df_filtered2['volum_negoci'].apply(lambda x: 1 if x > 0 else 0)
    # df_filtered2['rendiment_capital_mobiliari'] = df_filtered2['rendiment_capital_mobiliari'].apply(lambda x: 1 if x > 0 else 0)
    
    cols_to_drop2 = [
        'sectordereferencia', 'data_resolucio', 'data_adjudicacio', 'data_alta_ajut',
        'motiu_baixa', 'sobrevinguda', 'creuament_enviat', 'deduccio_infants_acolliment_familia_extensa_aliena',
        'alta_multiple', 'Barri'
    ]
    
    df_filtered3 = df_filtered2.drop(cols_to_drop2, axis=1)
    
    return df_filtered3
