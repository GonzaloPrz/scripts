#!/usr/bin/env bash
#############################################################################
# Script: train_and_bootstrap.sh
#
# Entrenar y hacer bootstrap
# de modelos con búsqueda bayesiana.
#############################################################################

# -- 1. Inicializar variables con valores por defecto
project_name=
feature_selection=1
filter_outliers=1
shuffle_labels=0
k=5
n_iter=15
init_points=20

# -- 2. Procesar argumentos
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -project_name)
            project_name="$2"
            shift # shift para avanzar al siguiente argumento
            shift
            ;;
        -feature_selection)
            feature_selection="$2"
            shift
            shift
            ;;
        -filter_outliers)
            filter_outliers="$2"
            shift
            shift
            ;;
        -shuffle_labels)
            shuffle_labels="$2"
            shift
            shift
            ;;
        -k)
            k="$2"
            shift
            shift
            ;;
        -n_iter)
            n_iter="$2"
            shift
            shift
            ;;
        -init_points)
            init_points="$2"
            shift
            shift
            ;;
        *)
            # Cualquier opción desconocida se ignora o podrías manejarla con un echo
            shift
            ;;
    esac
done

# -- 3. Verificar parámetro obligatorio: project_name
if [ -z "$project_name" ]; then
    echo "Error: el parámetro -project_name es obligatorio."
    exit 1
fi

# -- 5. Llamar a los scripts de Python
# Ajusta aquí la ruta a tus scripts, especialmente si estás en Windows o WSL
python "/Users/gp/scripts/shared/train_models_bayes.py" \
    "$project_name" "$feature_selection" "$filter_outliers" "$shuffle_labels" \
    "$k" "$n_iter" "$init_points"

python "/Users/gp/scripts/shared/bootstrap_models_bayes.py" \
    "$project_name" "$feature_selection" "$filter_outliers" "$shuffle_labels" \
    "$k"

# -- 6. Mostrar los parámetros usados
echo " "
echo "Pipeline executed with:"
echo "  project_name=$project_name"
echo "  feature_selection=$feature_selection"
echo "  filter_outliers=$filter_outliers"
echo "  shuffle_labels=$shuffle_labels"
echo "  k=$k"
echo "  n_iter=$n_iter"
echo "  init_points=$init_points"
echo " "