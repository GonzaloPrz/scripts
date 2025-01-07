import pandas as pd
from pathlib import Path
from warnings import simplefilter

# Ignorar FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

# Cargar el dataset
file_path = Path(Path.home(), 'data', 'MPLS', 'nps_features_data_MPLS_cat.csv')
data = pd.read_csv(file_path)

# Normalizar participant_code
data['id'] = data['id'].astype(str)
data.loc[data['id'] == '102.', 'id'] = '102'
data.loc[data['id'] == '103.', 'id'] = '103'

# Verificar duplicados iniciales
duplicates = data.duplicated(subset=['id'], keep=False).sum()
print(f"Duplicados iniciales en participant_code: {duplicates}")

# Identificar columnas de features y otros datos
features_columns = [col for col in data.columns if col.startswith('feature__')]
other_features = []
protocol_item_titles = data['protocol_item_title'].unique()

# Crear un DataFrame vacío para la transformación
transformed_data = pd.DataFrame()

# Iterar sobre columnas de features y valores únicos de protocol_item_title
for feature_col in features_columns:
    print(f"Procesando feature: {feature_col}")
    temp_data = pd.DataFrame()  # DataFrame temporal para cada feature
    
    for title in protocol_item_titles:
        if 'Bloque' in title:
            continue
        
        # Filtrar datos por título
        filtered_data = data[data['protocol_item_title'] == title].copy()
        filtered_data = filtered_data.dropna(subset=[feature_col])
        
        # Resolver duplicados por participant_code y feature_col
        filtered_data = filtered_data.drop_duplicates(subset=['id'])
        
        # Crear la nueva columna con el prefijo
        new_col_name = f"{title}_{feature_col}"
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
print(f"Duplicados finales en participant_code: {duplicates_after}")

# Guardar el DataFrame transformado en un archivo CSV
transformed_file_path = Path(Path.home(), 'data', 'MPLS', 'transformed_features_data_MPLS_cat.csv')
transformed_data.to_csv(transformed_file_path, index=False)

print(f"Dataset transformado guardado en: {transformed_file_path}")