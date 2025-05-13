import pandas as pd

def join_preprocessing(df_SIAD_new, df_beques):
    """
    Reads two dataframes from MENJADOR and SIAD and preprocesses them."
    """

    # INTEGRACIÓ. ID_UPC = ID_UPC_membreX
    """
    1. id_SIAD_members: membres del SIAD amb fills a beques
    2. Reshape de beques on cada fila és un membre
    3. Reducció de beques amb només els membres al SIAD
    4. Group by ID i agrupació de les variables
    """
    df_beques.rename(columns = {"ID_UPC_membre1": "Id_UPC_membre1"}, inplace = True)

    id_beques_members = set() # ID of all members and children in beques
    for i in range(11):
        membre = f'Id_UPC_membre{i}'
        if membre in df_beques.columns:
            id_beques_members.update(df_beques[membre].dropna().tolist())
    id_beques_members.update(df_beques["ID_UPC_sollicitant"].dropna().tolist())

    id_SIAD_members = set() # ID SIAD members with chindren in beques
    for _, row in df_SIAD_new.iterrows():
        if row['ID_UPC'] in id_beques_members:
            id_SIAD_members.add(row['ID_UPC'])

    df_beques_by_member = pd.melt( # Df beques where each member is one row
        df_beques,
        id_vars=[col for col in df_beques.columns if not col.startswith(('Id_UPC_membre', 'membre', 'Sexe_membre', 'Data_naix_membre', 'Id_UPC_memebre'))],
        value_vars=[f'Id_UPC_membre{i}' for i in range(11) if f'Id_UPC_membre{i}' in df_beques.columns],
        var_name='membre',
        value_name='ID_UPC'
    )

    df_beques_by_member = df_beques_by_member.dropna(subset=['ID_UPC'])
    df_beques_by_member_filtered = df_beques_by_member[df_beques_by_member['ID_UPC'].isin(id_SIAD_members)]

    aggregations = {
        'B_CURS': df_beques_by_member_filtered.groupby('ID_UPC')['curs'].first(),
        'B_NUM_FILLS': df_beques_by_member_filtered.groupby('ID_UPC')['ID_UPC_sollicitant'].size(), # Total de fills
        'B_RENDIMENT_CAPITAL_MOBILIARI': df_beques_by_member_filtered.groupby('ID_UPC')['rendiment_capital_mobiliari'].mean(),
        'B_VOLUM_NEGOCI': df_beques_by_member_filtered.groupby('ID_UPC')['volum_negoci'].mean(),
        'B_RENDA_FAMILIAR': df_beques_by_member_filtered.groupby('ID_UPC')['renda_familiar'].mean(),
        'B_FINQUES_URBANES': df_beques_by_member_filtered.groupby('ID_UPC')['finques_urbanes'].first(), # or mean
        'B_FINQUES_RUSTIQUES': df_beques_by_member_filtered.groupby('ID_UPC')['finques_rustiques'].first(),
        'B_PTS_FAM': df_beques_by_member_filtered.groupby('ID_UPC')['punts_valoracio_ambit_a'].mean(),
        'B_FAM_NOMBOSA_GENERAL' : df_beques_by_member_filtered.groupby('ID_UPC')['familia_nombrosa_general'].first(),
        'B_FAM_NOMBOSA_GENERAL' : df_beques_by_member_filtered.groupby('ID_UPC')['familia_nombrosa_general'].first(),
        'B_FAM_NOMBROSA_ESP' : df_beques_by_member_filtered.groupby('ID_UPC')['familia_nombrosa_especial'].first(),
        'B_FAM_MONOPARENTAL' : df_beques_by_member_filtered.groupby('ID_UPC')['familia_monoparental'].first(),
        'B_FAM_MONOPARENTAL_ESP' : df_beques_by_member_filtered.groupby('ID_UPC')['familia_monoparental_especial'].first(), # No correspon a especial+monoparental
        'B_PTS_FAM': df_beques_by_member_filtered.groupby('ID_UPC')['punts_valoracio_ambit_a'].mean(),
        'B_RISC_SOCIAL' : df_beques_by_member_filtered.groupby('ID_UPC')['risc_social'].first(),
        'B_HABITS_MENJAR_INADEQUATS' : df_beques_by_member_filtered.groupby('ID_UPC')['habits_menjar_inadequats'].first(),
        'B_TRANSTORNS_EMOCIONALS_FAM' : df_beques_by_member_filtered.groupby('ID_UPC')['trastorns_emocionals_familia'].first(),
        'B_PREU_AJUTS' : df_beques_by_member_filtered.groupby('ID_UPC')['preu_ajut'].sum()

    }

    for var, series in aggregations.items():
        df_SIAD_new[var] = df_SIAD_new['ID_UPC'].map(series) 
        
    max_children = int(df_beques_by_member_filtered.groupby('ID_UPC')['ID_UPC_sollicitant'].nunique().max()) # Maximim number of children

    # Group by member and its children variables (each variable is duplicated for each child)
    child_variables = {}
    variables = ['BECA_GARANTIDA', 'DISCAPACITAT_MES_33', 'PUNTS_UNITAT_FAM', 'NEGLIGENCIA_LLE',
                'SEGUIMENT_MEDIC_EXESSIU', 'CP', 'SEXE', 'DATA_NAIXEMENT', 'NACIONALITAT',
                'CENTRE_ESCOLAR', 'NIVELL_ESCOLAR', 'CURS']

    for var in variables:
        for i in range(1, max_children+1):
            var_name = f'B_{var}_fill_{i}'
            child_variables[var_name] = {}

    for id in id_beques_members:
        familia = df_beques_by_member_filtered[df_beques_by_member_filtered['ID_UPC'] == id]

        children_ids = familia['ID_UPC_sollicitant'].unique()

        for i, id_child in enumerate(children_ids, 1):  # Range from 1 to max num of children
            child = familia[familia['ID_UPC_sollicitant'] == id_child]
            child_variables[f'B_BECA_GARANTIDA_fill_{i}'][id] = child['beca_garantida'].iloc[0]
            child_variables[f'B_DISCAPACITAT_MES_33_fill_{i}'][id] = child['discapacitat_mes_33'].iloc[0]
            child_variables[f'B_PUNTS_UNITAT_FAM_fill_{i}'][id] = float(child['punts_unitat_familiar'].iloc[0])
            child_variables[f'B_NEGLIGENCIA_LLE_fill_{i}'][id] = child['negligencia_lleu'].iloc[0]
            child_variables[f'B_SEGUIMENT_MEDIC_EXESSIU_fill_{i}'][id] = child['seguiment_medic_excessiu'].iloc[0]
            child_variables[f'B_CP_fill_{i}'][id] = float(child['cp'].iloc[0])
            child_variables[f'B_SEXE_fill_{i}'][id] = child['sexe'].iloc[0]
            child_variables[f'B_DATA_NAIXEMENT_fill_{i}'][id] = int(child['datanaixement'].iloc[0])
            child_variables[f'B_NACIONALITAT_fill_{i}'][id] = child['nacionalitat'].iloc[0]
            child_variables[f'B_CENTRE_ESCOLAR_fill_{i}'][id] = float(child['centreescolar_id'].iloc[0])
            child_variables[f'B_NIVELL_ESCOLAR_fill_{i}'][id] = child['nivellescolar'].iloc[0]
            child_variables[f'B_CURS_fill_{i}'][id] = child['cursescolar'].iloc[0]

    for var_name, id_values in child_variables.items():
        df_SIAD_new[var_name] = df_SIAD_new['ID_UPC'].map(id_values)
        
    # Count the total number of citations by member
    SIAD_aggregations = {
    'TOTAL_CITES': df_SIAD_new.groupby('ID_UPC')['COD_CITA'].size()
    }

    for var, series in SIAD_aggregations.items():
        df_SIAD_new[var] = df_SIAD_new['ID_UPC'].map(series)
        

        # Identify which rows have actual scholarship data
    has_beca_data = df_SIAD_new['B_CURS'].notna()

    # Create the filtered dataset with only records that have scholarship data
    df_SIAD_filtered_final = df_SIAD_new[has_beca_data].copy()

    return df_SIAD_filtered_final