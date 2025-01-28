@echo off
:: Default values
set "project_name="
set "hyp_opt=1"
set "filter_outliers=0"
set "shuffle_labels=0"
set "stratify=1"
set "k=0"
set "n_iter=50"
set "n_iter_features=50"
set "feature_sample_ratio=0.5"
set "n_models=0"
set "feature_selection=1"

:: Process arguments
:process_args
:next_arg
if "%~1"=="" goto check_required
if "%~1"=="-project_name" (
    set "project_name=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-hyp_opt" (
    set "hyp_opt=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-filter_outliers" (
    set "filter_outliers=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-shuffle_labels" (
    set "shuffle_labels=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-stratify" (
    set "stratify=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-k" (
    set "k=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-n_iter" (
    set "n_iter=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-n_iter_features" (
    set "n_iter_features=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-feature_sample_ratio" (
    set "feature_sample_ratio=%~2"
    shift
    shift
    goto next_arg
)
if "%~1"=="-n_models" (
    set "n_models=%~2"
    shift
    shift
    goto next_arg
)
echo Error: Unknown argument %~1
goto show_help

:check_required
if "%project_name%"=="" (
    echo Error: The -project_name parameter is required.
    goto show_help
)

:: Configure feature selection
if "%n_iter_features%"=="0" (
    set "feature_selection=0"
)

call "C:\Users\CNC Audio\gonza\gonza-env\Scripts\activate"

:: Call Python scripts
::python "C:\Users\CNC Audio\gonza\scripts\shared\train_models.py" "%project_name%" "%hyp_opt%" "%filter_outliers%" "%shuffle_labels%" "%stratify%" "%k%" "%n_iter%" "%n_iter_features%" "%feature_sample_ratio%"
python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bca.py" "%project_name%" "%hyp_opt%" "%filter_outliers%" "%shuffle_labels%" "%feature_selection%" "%k%" "%n_models%"
python "C:\Users\CNC Audio\gonza\scripts\shared\test_models.py" "%project_name%" "%hyp_opt%" "%filter_outliers%" "%shuffle_labels%" "%k%"

:: Display used parameters
echo Pipeline executed with:
echo   project_name=%project_name%
echo   hyp_opt=%hyp_opt%
echo   filter_outliers=%filter_outliers%
echo   shuffle_labels=%shuffle_labels%
echo   stratify=%stratify%
echo   k=%k%
echo   n_iter=%n_iter%
echo   n_iter_features=%n_iter_features%
echo   feature_sample_ratio=%feature_sample_ratio%
echo   n_models=%n_models%