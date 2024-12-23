import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(Path.home(),'data','MPLS','MPLS_data.csv'))

orden = [
    'Estado General',
    'Animales',
    'Palabras con F',
    'Recuerdo feliz',
    'Fin Bloque 1',
    'Estado General 2',
    'Consulta sobre soledad 1',
    'Consulta sobre soledad 2',
    'Consulta sobre soledad 3',
    'Estimulos musicales',
    'Fin Bloque 2',
    'Estado General 3',
    'Nervios - Estrés',
    'Tristeza - (video)',
    'Fin Bloque 3'
]
# Define el orden estricto de los títulos de protocolo
orden_estricto = [
    "Estado General",
    "Animales",
    "Palabras con F",
    "Recuerdo feliz",
    "Fin Bloque 1",
    "Estado General 2",
    "Consulta sobre soledad 1",
    "Consulta sobre soledad 2",
    "Consulta sobre soledad 3",
    "Estimulos musicales",
    "Fin Bloque 2",
    "Estado General 3",
    "Nervios - Estrés",
    "Tristeza - (video)",
    "Fin Bloque 3"
]

# Ordenar los datos según `participant_code` y el orden estricto de `protocol_item_title`
data_ordenada = (
    df.assign(order=df['protocol_item_title'].map({v: i for i, v in enumerate(orden)}))
    .sort_values(by=['participant_code', 'order'])
    .drop(columns=['order'])
    .reset_index(drop=True)
)

data_ordenada.to_csv(Path(Path.home(),'data','MPLS','MPLS_data_ordenada.csv'), index=False)

columnas_numericas = [col for col in data_ordenada.columns if 'feature_' in col and not isinstance(data_ordenada[col][0], str)]

for col in columnas_numericas:
    data_ordenada[col] = data_ordenada[col].fillna(0)

# Tareas antes de "Estimulos musicales" excluyendo "Estimulos musicales"
mascara_pre_est = data_ordenada['protocol_item_title'].isin([
    "Estado General",
    "Animales",
    "Palabras con F",
    "Recuerdo feliz",
    "Estado General 2",
    "Consulta sobre soledad 1",
    "Consulta sobre soledad 2",
    "Consulta sobre soledad 3"
])

# Tareas después de "Estimulos musicales"
mascara_post_est = data_ordenada['protocol_item_title'].isin([
    "Estado General 3",
    "Nervios - Estrés",
    "Tristeza - (video)",
])

def procesar_est(data, mascara, sufijo):
    # Filtrar datos según la máscara
    bloque = data[mascara].copy()
    
    # Promediar las columnas numéricas para tareas repetidas
    bloque = bloque.groupby(['participant_code', 'protocol_item_title'])[columnas_numericas].mean().reset_index()
    
    # Renombrar las columnas numéricas
    for columna in columnas_numericas:
        bloque = bloque.rename(columns={
            columna: f"{columna}_{sufijo}"
        })
    
    # Crear un DataFrame con columnas renombradas para cada tarea
    bloque = bloque.pivot(index='participant_code', columns='protocol_item_title')
    bloque.columns = [
        f"{titulo}_{columna}_{sufijo}" for titulo, columna in bloque.columns
    ]
    bloque = bloque.reset_index()
    
    return bloque

# Procesar los datos pre y post estímulo musical
pre_est = procesar_est(data_ordenada, mascara_pre_est, "pre")
post_est = procesar_est(data_ordenada, mascara_post_est, "post")

# Combinar los bloques pre y post estímulo musical
resultado_est = pre_est.merge(post_est, on='participant_code', how='outer')
resultado_est.to_csv(Path(Path.home(),'data','MPLS','MPLS_data_renamed.csv'), index=False)

