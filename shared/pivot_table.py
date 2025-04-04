import pandas as pd
from pathlib import Path
from warnings import simplefilter

# Ignorar FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

# Cargar el dataset
file_path = Path(Path.home(), 'data', 'sj', 'pitch_analysis_features.csv')
data = pd.read_csv(file_path)

print(f"Number of participants: {data['id'].nunique()}")

all_data = pd.DataFrame()

# Iterar sobre columnas de features y valores únicos de protocol_item_title
# Identificar columnas de features y otros datos
features_columns = [col for col in data.columns if '__' in  col]
other_features = []
transformed_data = pd.DataFrame()

for feature_col in features_columns:
    #print(f"Procesando feature: {feature_col}")
    temp_data = pd.DataFrame()  # DataFrame temporal para cada feature
    tasks = data['task'].unique()

    for task in tasks:
        
        # Filtrar datos por título
        filtered_data = data[data['task'] == task].copy()
        filtered_data = filtered_data.dropna(subset=[feature_col])
        
        # Resolver duplicados por participant_code y feature_col
        filtered_data = filtered_data.drop_duplicates(subset=['id'])
        
        # Crear la nueva columna con el prefijo
        new_col_name = f"{task}__{feature_col}"
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
transformed_file_path = Path(Path.home(), 'data', 'sj', f'pitch_analysis_features.csv')
transformed_data.to_csv(transformed_file_path, index=False)

print(f"Dataset transformado guardado en: {transformed_file_path}")