import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import itertools

data = pd.read_csv(Path(Path.home(),'data','MPLS','reunion','all_data.csv'))

# Filtrar las columnas relevantes
columns_of_interest = [
    "emotion-analysis__joy",
    "emotion-analysis__sadness",
    'sentiment-analysis__POS',
    'sentiment-analysis__NEG',
]

protocols = [#"Consulta sobre soledad 1","Consulta sobre soledad 2", "Consulta sobre soledad 3",
             'Estado General','Estado General 2','Estado General 3']

for protocol in protocols:
    data_protocol = {'protocol_item_title':protocol}
    data_protocol.update(dict((col,data[f'{protocol}_{col}']) for col in columns_of_interest))
    data_protocol = pd.DataFrame(data_protocol)
    if protocol == protocols[0]:
        filtered_data = data_protocol
    else:
        filtered_data = pd.concat([filtered_data, data_protocol])

#Calcular valores medios e intervalos de confianza
mean_values = filtered_data.groupby("protocol_item_title").mean()
std_values = filtered_data.groupby("protocol_item_title").std()
n_values = filtered_data.groupby("protocol_item_title").count()
std_error_values = std_values / n_values ** 0.5

# Graficar los resultados
plt.figure(figsize=(10, 6))
mean_values.plot(kind="line", marker="o", figsize=(10, 6))
plt.title("Modificación del valor medio de análisis de sentimiento", fontsize=14)
plt.xlabel("Protocolos", fontsize=12)
plt.ylabel("Valor medio", fontsize=12)
plt.xticks(range(len(protocols)), protocols, fontsize=10)
plt.legend(title="Variables", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig(Path(Path.home(),'data','MPLS','sentiment_analysis_estado_general.png'))

from scipy.stats import f_oneway

# Preparar los datos para el ANOVA
anova_results = {}
for variable in columns_of_interest:
    # Obtener los valores por protocolo
    groups = [
        filtered_data[filtered_data["protocol_item_title"] == protocol][variable].dropna()
        for protocol in protocols
    ]
    # Realizar el ANOVA
    stat, p_value = f_oneway(*groups)
    anova_results[variable] = {"F-statistic": stat, "p-value": p_value}

anova_results