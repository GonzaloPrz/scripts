@echo off

:: Check if the user provided the required project_name argument
if "%1"=="" (
    echo Usage: run_pipeline.bat project_name [hyp_opt] [filter_outliers] [shuffle_labels] [stratify] [k] [n_iter] [n_iter_features] [feature_sample_ratio] [n_boot] [boot_test] [scaler_name] [id_col] [n_seeds_train]
    echo Optional parameters:
    echo   hyp_opt         - Set to 1 for True, 0 for False (default: 1)
    echo   filter_outliers  - Set to 1 for True, 0 for False (default: 0)
    echo   shuffle_labels   - Set to 1 for True, 0 for False (default: 0)
    echo   stratify        - Set to 1 for True, 0 for False (default: 0)
    echo   k             - Number of folds in cross-validation (default: 5)
    echo   n_iter          - Number of combination of hyperparameters to test (default: 50)
    echo   n_iter_features - Number of features to test (default: 50)
    echo  feature_sample_ratio - Ratio of features to sample in each bootstrap sample (default: 0.5)
    echo   n_boot          - Number of bootstrap samples in dev (default: 200)
    echo   boot_test       - Number of bootstrap samples in test (default: 200)
    echo scaler_name     - Name of the scaler to use (default: StandardScaler)
    echo id_col          - Name of the column with the unique identifier (default: id)
    echo n_seeds_train   - Number of seeds to use in train (default: 10)

    exit /b 1
)

:: Required parameter
set project_name=%1

:: Optional parameters with default values
set hyp_opt = 1
set filter_outliers=0
set shuffle_labels=0
set stratify=0
set k=5
set n_iter=50
set n_iter_features=50
set feature_sample_ratio=0.5
set n_boot=200
set boot_test=200
set scaler_name  = StandardScaler
set id_col = id
set n_seeds_train=10
set n_seeds_test=1
set feature_selection = 1

if NOT "%2"=="" set hyp_opt=%2
if NOT "%3"=="" set filter_outliers=%3
if NOT "%4"=="" set shuffle_labels=%4
if NOT "%5"=="" set stratify=%5
if NOT "%6"=="" set k=%6
if NOT "%7"=="" set n_iter=%7
if NOT "%8"=="" set n_iter_features=%8
if NOT "%9"=="" set feature_sample_ratio=%9
if NOT "%10"=="" set n_boot=%10
if NOT "%11"=="" set boot_test=%11
if NOT "%12"=="" set scaler_name=%12
if NOT "%13"=="" set id_col=%13
if NOT "%14"=="" set n_seeds_train=%14
if n_iter_features == 0 set feature_selection = 0

:: Call your Python scripts and pass all parameters
python "C:\Users\CNC Audio\gonza\scripts\train_models.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %stratify% %k% %n_iter% %n_iter_features% %feature_sample_ratio% %scaler_name% %id_col% %n_seeds_train%
python "C:\Users\CNC Audio\gonza\scripts\bootstrap_models_bca.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %feature_selection% %k% %n_boot% %scaler_name% %id_col% 
python "C:\Users\CNC Audio\gonza\scripts\test_models.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %k% %n_boot_test% %scaler_name% %id_col%

echo Pipeline executed with:
echo   project_name=%project_name%
echo  hyp_opt=%hyp_opt%
echo   filter_outliers=%filter_outliers%
echo   shuffle_labels=%shuffle_labels%
echo   stratify=%stratify%
echo   k=%k%
echo   n_iter=%n_iter%
echo   n_iter_features=%n_iter_features%
echo   feature_sample_ratio=%feature_sample_ratio%
echo   n_boot=%n_boot%
echo   boot_test=%boot_test%
echo   scaler_name=%scaler_name%
echo   id_col=%id_col%
echo  n_seeds_train=%n_seeds_train%

pause