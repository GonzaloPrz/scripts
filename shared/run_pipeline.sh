#!/bin/bash

# Valores por defecto
project_name=""
hyp_opt=1
filter_outliers=1
shuffle_labels=0
k=5
n_iter=50
n_iter_features=50
feature_sample_ratio=0.5
feature_selection=1

# Función para mostrar ayuda
function show_help {
    echo "Uso: ./run_pipeline.sh -project_name <nombre_proyecto> [opciones]"
    echo "Opciones:"
    echo "  -project_name           Nombre del proyecto (obligatorio)"
    echo "  -hyp_opt                Habilitar optimización de hiperparámetros (default: 1)"
    echo "  -filter_outliers        Filtrar valores atípicos (default: 1)"
    echo "  -shuffle_labels         Barajar etiquetas (default: 0)"
    echo "  -k                      Valor de k para validación cruzada (default: 5)"
    echo "  -n_iter                 Número de iteraciones (default: 50)"
    echo "  -n_iter_features        Número de iteraciones para selección de características (default: 50)"
    echo "  -feature_sample_ratio   Proporción de muestra para selección de características (default: 0.5)"
    exit 1
}

# Procesar argumentos
while [[ $# -gt 0 ]]; do
    case "$1" in
        -project_name)
            project_name="$2"
            shift 2
            ;;
        -hyp_opt)
            hyp_opt="$2"
            shift 2
            ;;
        -filter_outliers)
            filter_outliers="$2"
            shift 2
            ;;
        -shuffle_labels)
            shuffle_labels="$2"
            shift 2
            ;;
        -stratify)
            stratify="$2"
            shift 2
            ;;
        -k)
            k="$2"
            shift 2
            ;;
        -n_iter)
            n_iter="$2"
            shift 2
            ;;
        -n_iter_features)
            n_iter_features="$2"
            shift 2
            ;;
        -feature_sample_ratio)
            feature_sample_ratio="$2"
            shift 2
            ;;
        *)
            echo "Error: argumento desconocido $1"
            show_help
            ;;
    esac
done

# Verificar parámetros obligatorios
if [[ -z "$project_name" ]]; then
    echo "Error: el parámetro -project_name es obligatorio."
    show_help
fi

# Configurar selección de características
if [[ "$n_iter_features" -eq 0 ]]; then
    feature_selection=0
fi

# Llamar a los scripts de Python
python "/Users/gp/scripts/shared/train_models.py" "$project_name" "$hyp_opt" "$filter_outliers" "$shuffle_labels" "$stratify" "$k" "$n_iter" "$n_iter_features" "$feature_sample_ratio"
python "/Users/gp/scripts/shared/bootstrap_models_bca.py" "$project_name" "$hyp_opt" "$filter_outliers" "$shuffle_labels" "$feature_selection" "$k"
python "/Users/gp/scripts/shared/test_models.py" "$project_name" "$hyp_opt" "$filter_outliers" "$shuffle_labels" "$k"

# Mostrar los parámetros usados
echo "Pipeline executed with:"
echo "  project_name=$project_name"
echo "  hyp_opt=$hyp_opt"
echo "  filter_outliers=$filter_outliers"
echo "  shuffle_labels=$shuffle_labels"
echo "  k=$k"
echo "  n_iter=$n_iter"
echo "  n_iter_features=$n_iter_features"
echo "  feature_sample_ratio=$feature_sample_ratio"