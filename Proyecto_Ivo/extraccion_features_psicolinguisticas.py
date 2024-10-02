
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:57:57 2022

@author: fferrante
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

#import Estadistica as estadistica

from scipy.stats import kurtosis
from scipy.stats import skew

sys.path.append(str(Path(Path(__file__).parent.parent,'NLP-Labo')))

import NLPClass

import re      # libreria de expresiones regulares

import math
import itertools

def porcentaje_distribucion_palabras(df):
    df_count = pd.DataFrame(data=None, columns=df.columns)

    for i,row in df.iterrows():
        porcentajes = []
    
        for column in df.columns:
            porcentajes.append(len(df.at[i,column]))
        df_length = len(df_count)
        df_count.loc[df_length] = porcentajes
        
    df_count = df_count.div(df_count.sum(axis=1), axis=0)
    return df_count

def list_to_txt_file(name,lista, separator = "\n"):
    textfile = open(name, "w")
    for element in lista:
        textfile.write(element + separator)
    textfile.close()

# Defino una funcion que recibe un texto y devuelve el mismo texto sin signos,
def clean_text( text, char_replace = ''):
    
    # pasa las mayusculas del texto a minusculas
    text = text.lower()                                              
    # Conservo solo caracteres alfanuméricos y guión
    text = re.sub('[^a-zA-Z0-9 \náéíóúÁÉÍÓÚñÑü]', char_replace, text)
    return text
# %% Leemos el dataframe

cwd = Path(__file__).parent
transcripts_filename = 'Transcripts.xlsx'
# Cargo el excel en dataframes
df_original = pd.read_excel(Path(cwd,'data',transcripts_filename))

columnas = ['letra_f',
            'letra_a','letra_s','animales'
            ]

# %% Extraigo features psicolinguisticas

nlp_class = NLPClass.NLPClass()

# Reemplazo los nans por vacío
df_original.fillna('', inplace=True)

# Limpio el texto, quedándome solo con los caracteres alfanuméricos
for columna in columnas:
    df_original[columna] = [clean_text(linea,' ') for linea in df_original[columna].values]

# Separo por guión las palabras
for columna in columnas:
    df_original[columna] = [linea.split("-") for linea in df_original[columna].values]

for columna in columnas:
    for r, row in df_original.iterrows(): 
        for columna in columnas:
            df_original.at[r,columna] = str(row[columna]).replace('[','').replace(']','').replace("'","").replace(',','').split()
  
# Elimino elementos vacíos y reemplazo los dobles espacios por espacios simples
for columna in columnas: 
    for i_row, row in df_original.iterrows():
        df_original.at[i_row,columna] = [palabra.replace("  "," ") for palabra in row[columna] if palabra != "" and palabra != " "]

for columna in columnas:
    for i_row, row in df_original.iterrows():
        df_original.at[i_row,columna] = [palabra for palabra in row[columna] if ((palabra != "xxx") and (palabra != "palabra incomprensible"))]

# Obtengo los tokens únicos y la cantidad de veces que aparece cada uno.
unique_words_list = []
count_words_list = []
for columna in columnas:
    unique_words, count_words = nlp_class.count_words(df_original[columna])
    unique_words_list.append(unique_words)
    count_words_list.append(count_words)
    
path = Path(Path(__file__).parent.parent,'NLP-Labo','translations.pkl')

unique_total_words = []
for list_palabras in unique_words_list:
    unique_total_words = unique_total_words + list_palabras

# %%
unique_total_words = np.unique(unique_total_words)
nlp_class.add_to_pickle_translation_file(path,unique_total_words,lan_src = "spanish",lan_dest = "english")

# %% Leo el archivo con las psicolinguisticas y hago la extracción para los textos de esta base

data = pd.read_csv(Path(Path.home(),'NLP-Labo','psycho_spanish.csv'), encoding='utf-8')
columnas_psicolinguisticas = ["frequency",
                              "num_phonemes","num_syll","phon_neigh"
                              ]

df_original = nlp_class.psycholinguistics_features_synonyms(data, columnas_psicolinguisticas, columnas, df_original,str(path))

# %% Cantidad de nans

# Función para contar valores no NaN en una lista

count_non_nan = lambda lista: sum(~np.isnan(lista))

for columna,columna_psico in itertools.product(columnas,columnas_psicolinguisticas):
    df_original[columna + "_" + columna_psico + "_porc_nans"] = np.nan

    for i_row, row in df_original.iterrows():
        if len(row[columna + "_" + columna_psico]) > 0:
               df_original.at[i_row,columna + "_" + columna_psico + "_porc_nans"] = (sum(math.isnan(x) for x in row[columna + "_" + columna_psico])*100) / len(row[columna + "_" + columna_psico])
    # Aplicar la función a la columna del DataFrame
    df_original[columna + "_" + columna_psico + "_cant_valores"] = df_original[columna + "_" + columna_psico].apply(count_non_nan)

# %% Calculo el promedio, mínimo, máximo, std, mediana, curtosis y skewness 
# para cada variable psicolingüística para cada columna de fluencia y las 
# agrego al dataframe.

list_statistics = ["promedio","minimo","maximo","std","mediana","curtosis","skewness"]
df_original = nlp_class.obtain_stadistic_values(df_original, columnas_psicolinguisticas, columnas, list_statistics)

# %% Cargo el modelo de fast_text
path_fast = str(Path(Path.home(),'NLP-Labo','cc.es.300.bin'))

#nlp_class.load_fast_text_model(path_fast)

# %% Obtengo el word_embedding por cada palabra

#for column in columnas:
#    df_original[column+"_embedding"] = np.nan
#    df_original[column+"_embedding"] = df_original[column+"_embedding"].astype(object)
    
#    for i,row in df_original.iterrows():
#        df_original.at[i,column+"_embedding"] = [nlp_class.get_word_fast_text_vector([[word]])[0][0] for word in row[column]]
       
# %% calculo la ongoing semantic variability

# Levanto el archivo con todas las traducciones hechas

#for columna in columnas:
#    df_original[columna + "_osv"] = nlp_class.ongoing_semantic_variability_complete(df_original[columna].values)
  
# %% Calculo la granularidad semántica

path = Path(Path.home(),'NLP-Labo','translations.pkl')

df_translated = nlp_class.read_pickle_translation_file(path)

maxima_granularidad = -2

def count_non_nan_values(lst):
    count = 0
    for value in lst:
        if not math.isnan(value):
            count += 1
    return count

for columna in columnas:

    df_original[columna + "_granularidad"] = np.nan
    df_original[columna + "_granularidad"] = df_original[columna+"_granularidad"].astype(object)
    df_original[columna + "_granularidad"+"_promedio"] = np.nan
    df_original[columna + "_granularidad"+"_minimo"] = np.nan
    df_original[columna + "_granularidad"+"_maximo"] = np.nan
    df_original[columna + "_granularidad"+"_std"] = np.nan
    df_original[columna + "_granularidad"+"_mediana"] = np.nan
    df_original[columna + "_granularidad"+"_curtosis"] = np.nan
    df_original[columna + "_granularidad"+"_skewness"] = np.nan
    df_original[columna + "_granularidad"+"_porc_nans"] = np.nan

    for i,row in df_original.iterrows():
        lista = []
        palabras_sustantivos = []
        for palabra in row[columna]:
            palabra = palabra.replace(',','').replace("'","")

            translation = df_translated[(df_translated["word"] == palabra) &
                                        (df_translated["lan_src"] == "spanish") &
                                        (df_translated["lan_dest"] == "english")]
            if (len(translation.index)>1):
                print("Hay mas de una traduccion para esa palabra")
            else:
                posibles_traducciones = []
                if translation['translation'].values.shape[0] != 0:
                    if translation["translation"].values[0][1][0][0][5][0][4] is not None:
                        for j in range(0,len(translation["translation"].values[0][1][0][0][5][0][4])):
                            posibles_traducciones.append(translation["translation"].values[0][1][0][0][5][0][4][j][0].lower())
                        
                granularidad = nlp_class.min_nodes_distance_all_words(posibles_traducciones,hypernym_check="entity.n.01")
                lista.append(granularidad)
                if granularidad != -1:
                    palabras_sustantivos.append(palabra)

        if (len(lista)>0):
            df_original.at[i,columna + "_granularidad"+"_porc_nans"] = (len([x for x in lista if x == -1])*100) / len(lista)
            df_original.at[i,columna + "_granularidad"+"_cant_valores"] = count_non_nan_values(lista)

        else:
            df_original.at[i,columna + "_granularidad"+"_porc_nans"] = 0
            df_original.at[i,columna + "_granularidad"+"_cant_valores"] = 0


        lista = [x for x in lista if x != -1]
        if len(lista)>0:
            if np.max(lista)>maxima_granularidad:
                maxima_granularidad = np.max(lista)
        df_original.at[i,columna + "_granularidad"] = lista
        df_original.at[i,columna + "_granularidad"+"_promedio"] = np.nanmean(lista)
        if (len(lista)==0):
            df_original.at[i,columna + "_granularidad"+"_minimo"] = np.nan
            df_original.at[i,columna + "_granularidad"+"_maximo"] = np.nan
        else:
            df_original.at[i,columna + "_granularidad"+"_minimo"] = np.nanmin(lista)
            df_original.at[i,columna + "_granularidad"+"_maximo"] = np.nanmax(lista)
        df_original.at[i,columna + "_granularidad"+"_std"] = np.nanstd(lista)
        df_original.at[i,columna + "_granularidad"+"_mediana"] = np.nanmedian(lista)
        df_original.at[i,columna + "_granularidad"+"_curtosis"] = kurtosis([x for x in lista if str(x) != 'nan'])
        df_original.at[i,columna + "_granularidad"+"_skewness"] = skew([x for x in lista if str(x) != 'nan'])

# %% Guardar features psicolinguisticas de los participantes con más de 2 valores en todas las features

df_features_mayor_2 = df_original.copy(deep=True)


# Obtener los nombres de las columnas que terminan en "_cant_valores"
#columnas_cant_valores = [col for col in df_features_mayor_2.columns if col.endswith('_cant_valores')]

# Filtrar las filas con valores menores a 3 en las columnas seleccionadas
#df_features_mayor_2 = df_features_mayor_2[(df_features_mayor_2[columnas_cant_valores] >= 3).all(axis=1)]

features_list = ["promedio","minimo","maximo","std","mediana","curtosis","skewness","osv"]

# Filtrar las columnas que contengan alguno de los strings de la lista
columnas_filtradas = [col for col in df_features_mayor_2.columns if any(string in col for string in features_list)]

# Crear un nuevo dataframe con las columnas filtradas
df_features_mayor_2 = df_features_mayor_2[columnas_filtradas + ["Codigo"]]

cwd = Path(__file__).parent

Path(Path(__file__).parent,'data','bases_imputacion_sinonimos','bases_mayor_a_2','psicolinguisticas').mkdir(parents=True, exist_ok=True)

df_features_mayor_2.to_excel(Path(Path(__file__).parent,'data','bases_imputacion_sinonimos','bases_mayor_a_2','psicolinguisticas','psych_granularity.xlsx'),index=False)

# %% Guardar datos sociodemográficos de los participantes que tienen más de 2 valores en todas sus variables
'''
df_features_mayor_2_socio = df_features_mayor_2[["Codigo"]].merge(df_original)
columnas_demo = ["PatientID","Age","Education","Sex","GROUP"]
df_features_mayor_2_socio = df_features_mayor_2_socio[columnas_demo]
df_features_mayor_2_socio.to_csv(cwd + "//bases_imputacion_sinonimos//bases_mayor_a_2//demograficas//demografia_pd.csv")
'''














