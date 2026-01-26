import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Datos de AUC
auc_values = [
    [0.7, 0.75, 0.72, 0.76],  # Affective prosody (bvFTD vs HC), Affective prosody (AD vs HC), 
                              # Whole-audio prosody (bvFTD vs HC), Whole-audio prosody (AD vs HC)
]

# Etiquetas de los ejes
categories = ['Affective Prosody (bvFTD vs HC)', 'Affective Prosody (AD vs HC)',
              'Whole-audio Prosody (bvFTD vs HC)', 'Whole-audio Prosody (AD vs HC)']

# Número de categorías
num_vars = len(categories)

# Configuración de los valores de la gráfica
values = auc_values[0]
values += values[:1]  # Para cerrar el gráfico (volver al primer valor)

# Ángulos para cada categoría
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Gráfico de radar
fig, ax = plt.subplots(figsize=(6, 6), dpi=100, subplot_kw=dict(polar=True))

# Dibujar el gráfico
ax.fill(angles, values, color='blue', alpha=0.25)
ax.plot(angles, values, color='blue', linewidth=2)

# Etiquetas
ax.set_yticks([0.5, 0.6, 0.7, 0.8])
ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8'])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, fontweight='bold', color='black')

# Establecer el rango del gráfico (desde 0.5 hasta 0.8)
ax.set_ylim(0.5, 0.8)

# Estética del gráfico
ax.set_facecolor('white')
ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

# Títulos y ajustes finales
plt.title('AUC Comparison in Affective and Whole-audio Prosody', size=16, color='black', fontweight='bold')

# Mostrar gráfico
plt.show()