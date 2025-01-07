import pandas as pd
from pathlib import Path
from warnings import simplefilter

# Ignorar FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

# Cargar el dataset
file_path = Path(Path.home(), 'data', 'AKU', 'AKU_data_complete.csv')
data = pd.read_csv(file_path)

data = data[data['group'] == 0]

print(f"Number of participants: {data['id'].nunique()}")

# Normalizar participant_code
languages = data['language'].unique()
# Crear un DataFrame vacío para la transformación
all_data = pd.DataFrame()

# Iterar sobre columnas de features y valores únicos de protocol_item_title
for language in languages:
    
    data_lang = data[data['language'] == language]
    data_lang['id'] = data_lang['id'].astype(str)

    # Identificar columnas de features y otros datos
    features_columns = [col for col in data_lang.columns if '__' in  col]
    other_features = []
    transformed_data = pd.DataFrame()

    for feature_col in features_columns:
        #print(f"Procesando feature: {feature_col}")
        temp_data = pd.DataFrame()  # DataFrame temporal para cada feature
        tasks = data_lang['task'].unique()

        for task in tasks:
            
            # Filtrar datos por título
            filtered_data = data_lang[data_lang['task'] == task].copy()
            filtered_data = filtered_data.dropna(subset=[feature_col])
            
            # Resolver duplicados por participant_code y feature_col
            filtered_data = filtered_data.drop_duplicates(subset=['id'])
            
            # Crear la nueva columna con el prefijo
            new_col_name = f"{task}_{feature_col}"
            filtered_data[new_col_name] = filtered_data[feature_col]
            #filtered_data.rename(columns={feature_col: new_col_name}, inplace=True)
            
            # Combinar los datos
            if temp_data.empty:
                temp_data = filtered_data[['id',new_col_name] + other_features]
            else:
                temp_data = pd.merge(temp_data, filtered_data[['id',new_col_name] + other_features], on='id', how='outer')
        
        # Combinar datos procesados con el DataFrame transformado principal
        if transformed_data.empty:
            transformed_data = temp_data
        else:
            transformed_data = pd.merge(transformed_data, temp_data, on='id', how='outer')

    # Verificar duplicados finales
    duplicates_after = transformed_data.duplicated(subset=['id'], keep=False).sum()
    print(f"Duplicados finales en id: {duplicates_after}")

    # Guardar el DataFrame transformado en un archivo CSV
    transformed_file_path = Path(Path.home(), 'data', 'AKU', f'AKU_data_HC_{language}.csv')
    transformed_data.to_csv(transformed_file_path, index=False)

    print(f"Dataset transformado guardado en: {transformed_file_path}")

    if all_data.empty:
        all_data = transformed_data
    else:
        all_data = pd.concat((all_data, transformed_data), axis=0)
    
all_data.to_csv(Path(Path.home(), 'data', 'AKU', 'AKU_data_HC_all.csv'), index=False)