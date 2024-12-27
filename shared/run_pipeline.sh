#!/bin/bash

# Usage message
if [ -z "$1" ]; then
    echo "Usage: run_pipeline.sh project_name [hyp_opt] [filter_outliers] [shuffle_labels] [stratify] [k] [n_iter] [n_iter_features] [feature_sample_ratio] [n_boot] [boot_test] [scaler_name] [id_col] [n_seeds_train]"
    echo "Optional parameters:"
    echo "  hyp_opt             - Set to 1 for True, 0 for False (default: 1)"
    echo "  filter_outliers     - Set to 1 for True, 0 for False (default: 0)"
    echo "  shuffle_labels      - Set to 1 for True, 0 for False (default: 0)"
    echo "  stratify            - Set to 1 for True, 0 for False (default: 0)"
    echo "  k                   - Number of folds in cross-validation (default: 5)"
    echo "  n_iter              - Number of hyperparameter combinations to test (default: 50)"
    echo "  n_iter_features     - Number of features to test (default: 50)"
    echo "  feature_sample_ratio- Ratio of features to sample in each bootstrap sample (default: 0.5)"
    echo "  n_boot              - Number of bootstrap samples in dev (default: 200)"
    echo "  boot_test           - Number of bootstrap samples in test (default: 200)"
    echo "  scaler_name         - Name of the scaler to use (default: StandardScaler)"
    echo "  id_col              - Name of the column with the unique identifier (default: id)"
    echo "  n_seeds_train       - Number of seeds to use in train (default: 10)"
    exit 1
fi

# Required parameter
project_name=$1

# Optional parameters with default values
hyp_opt=${2:-1}
filter_outliers=${3:-0}
shuffle_labels=${4:-0}
stratify=${5:-0}
k=${6:-5}
n_iter=${7:-50}
n_iter_features=${8:-50}
feature_sample_ratio=${9:-0.5}
n_boot=${10:-200}
boot_test=${11:-200}
scaler_name=${12:-StandardScaler}
id_col=${13:-id}
n_seeds_train=${14:-10}
n_seeds_test=1
feature_selection=1

# Check for feature selection
if [ "$n_iter_features" -eq 0 ]; then
    feature_selection=0
fi

# Call Python scripts with all parameters
python3 "/path/to/scripts/train_models.py" "$project_name" "$hyp_opt" "$filter_outliers" "$shuffle_labels" "$stratify" "$k" "$n_iter" "$n_iter_features" "$feature_sample_ratio" "$scaler_name" "$id_col" "$n_seeds_train"
python3 "/path/to/scripts/bootstrap_models_bca.py" "$project_name" "$hyp_opt" "$filter_outliers" "$shuffle_labels" "$feature_selection" "$k" "$n_boot" "$scaler_name" "$id_col"
python3 "/path/to/scripts/test_models.py" "$project_name" "$hyp_opt" "$filter_outliers" "$shuffle_labels" "$k" "$boot_test" "$scaler_name" "$id_col"

# Output parameters for debugging
echo "Pipeline executed with:"
echo "  project_name=$project_name"
echo "  hyp_opt=$hyp_opt"
echo "  filter_outliers=$filter_outliers"
echo "  shuffle_labels=$shuffle_labels"
echo "  stratify=$stratify"
echo "  k=$k"
echo "  n_iter=$n_iter"
echo "  n_iter_features=$n_iter_features"
echo "  feature_sample_ratio=$feature_sample_ratio"
echo "  n_boot=$n_boot"
echo "  boot_test=$boot_test"
echo "  scaler_name=$scaler_name"
echo "  id_col=$id_col"
echo "  n_seeds_train=$n_seeds_train"