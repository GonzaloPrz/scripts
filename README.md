# Repositorio de Modelos y Evaluación

Este repositorio contiene scripts relacionados con el entrenamiento, evaluación y análisis de modelos de aprendizaje automático. Además, utiliza funciones definidas en el script `utils.py` del repositorio [Scripts Generales](https://github.com/GonzaloPrz/scripts_generales).

## Estructura del Repositorio

- **`train_models.py`**: Realiza el entrenamiento de modelos utilizando diversos algoritmos y configuraciones.
- **`test_models.py`**: Evalúa el rendimiento de los modelos entrenados en conjuntos de datos de prueba.
- **`bootstrap_models_bca.py`**: Implementa remuestreo (bootstrap) para estimar intervalos de confianza con corrección de sesgo.
- **`report_best_models.py`**: Genera reportes de los mejores modelos basados en sus métricas de evaluación.
- **`train_models_bayes.py`**: Optimiza los hiperparámetros de los modelos utilizando técnicas de optimización bayesiana.
- **`bootstrap_models_bayes.py`**: Realiza remuestreo para modelos entrenados con parámetros optimizados mediante métodos bayesianos.

---

## Descripción de los Scripts

### 1. `train_models.py`
Entrena modelos de aprendizaje automático para tareas de clasificación y regresión. Funcionalidades principales:

- Utiliza validación cruzada estratificada para entrenamiento.
- Admite varios algoritmos, incluidos:
  - Regresión logística.
  - Máquinas de soporte vectorial.
  - Modelos basados en vecinos más cercanos.
  - Modelos XGBoost.
- Configuración de escaladores, imputadores y métricas de evaluación.
- Almacena modelos entrenados y sus métricas en directorios específicos para análisis posterior.

### 2. `test_models.py`
Este script evalúa los modelos entrenados en conjuntos de datos de prueba:

- Divide los datos en conjuntos de entrenamiento y prueba.
- Calcula métricas de evaluación relevantes como AUC, precisión y error cuadrático medio.
- Genera reportes detallados de las predicciones y los valores reales.

### 3. `bootstrap_models_bca.py`
Calcula intervalos de confianza mediante bootstrap con corrección de sesgo:

- Realiza remuestreo con reemplazo.
- Evalúa modelos en múltiples muestras de bootstrap.
- Calcula intervalos de confianza para métricas como AUC, R² y error medio absoluto.

### 4. `report_best_models.py`
Genera reportes detallados de los mejores modelos seleccionados:

- Analiza los resultados almacenados para identificar el mejor modelo según una métrica específica.
- Produce reportes en formato legible para compartir con equipos o incluir en documentación.

### 5. `train_models_bayes.py`
Optimiza hiperparámetros de modelos utilizando optimización bayesiana:

- Define espacios de búsqueda para cada hiperparámetro.
- Admite múltiples algoritmos de aprendizaje y tareas (clasificación/regresión).
- Genera configuraciones óptimas para mejorar el rendimiento del modelo.

### 6. `bootstrap_models_bayes.py`
Aplica bootstrap para modelos optimizados mediante métodos bayesianos:

- Evalúa los modelos con diferentes muestras generadas por remuestreo.
- Calcula intervalos de confianza para los modelos ajustados.
- Permite validar la robustez de las configuraciones seleccionadas.

---

## Requisitos Previos

- **Python 3.8+**

Para instalar las dependencias:
```bash
pip install -r requirements.txt
