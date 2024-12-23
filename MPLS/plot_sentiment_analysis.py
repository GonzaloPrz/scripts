import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

data = pd.read_csv(Path(Path.home(),'data','MPLS','nps_features_data_MPLS.csv'))

# Filtrar las columnas relevantes
columns_of_interest = [
    "participant_code",
    "protocol_item_title",
    "feature__sentiment-analysis-gonza__POS",
    "feature__sentiment-analysis-gonza__NEU",
    "feature__sentiment-analysis-gonza__NEG",
]

protocols = ["Estado General 2", "Estado General 3"]

filtered_data = data[columns_of_interest]

participants_with_all_protocols = (
    filtered_data.groupby("participant_code")["protocol_item_title"]
    .apply(lambda x: set(x) == set(protocols))
)
valid_participants = participants_with_all_protocols[participants_with_all_protocols].index

# Filtrar los datos para incluir solo estos participantes
filtered_data_valid = filtered_data[filtered_data["participant_code"].isin(valid_participants)]

# Filtrar solo los protocolos de interés
filtered_data = filtered_data[filtered_data["protocol_item_title"].isin(protocols)]

#Rename columns
filtered_data.rename(columns={"feature__sentiment-analysis-gonza__POS": "POS",
                              "feature__sentiment-analysis-gonza__NEU": "NEU",
                              "feature__sentiment-analysis-gonza__NEG": "NEG"}, inplace=True)

# Agrupar por protocolo y calcular el valor medio de las variables
grouped_data = (
    filtered_data.groupby("protocol_item_title")[
        [
            "POS",
            "NEU",
            "NEG",
        ]
    ]
    .mean()
    .reindex(protocols)  # Ordenar según los protocolos
)

# Graficar los resultados
plt.figure(figsize=(10, 6))
grouped_data.plot(kind="line", marker="o", figsize=(10, 6))
plt.title("Modificación del valor medio de análisis de sentimiento", fontsize=14)
plt.xlabel("Protocolos", fontsize=12)
plt.ylabel("Valor medio", fontsize=12)
plt.xticks(range(len(protocols)), protocols, fontsize=10)
plt.legend(title="Variables", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

from scipy.stats import f_oneway

# Preparar los datos para el ANOVA
anova_results = {}
for variable in [
    "POS",
    "NEU",
    "NEG",
]:
    # Obtener los valores por protocolo
    groups = [
        filtered_data[filtered_data["protocol_item_title"] == protocol][variable].dropna()
        for protocol in protocols
    ]
    # Realizar el ANOVA
    stat, p_value = f_oneway(*groups)
    anova_results[variable] = {"F-statistic": stat, "p-value": p_value}

anova_results