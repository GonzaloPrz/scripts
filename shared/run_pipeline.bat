@echo off

:: Inicializar variables con valores por defecto
set project_name=""
set hyp_opt=1
set filter_outliers=1
set shuffle_labels=0
set k=5
set n_iter=50
set n_iter_features=50
set feature_sample_ratio=0.5
set feature_selection=1

:: Procesar argumentos
:parse_args
if "%~1"=="" goto end_args
if "%~1"=="-project_name" set project_name=%~2 & shift & shift & goto parse_args
if "%~1"=="-hyp_opt" set hyp_opt=%~2 & shift & shift & goto parse_args
if "%~1"=="-filter_outliers" set filter_outliers=%~2 & shift & shift & goto parse_args
if "%~1"=="-shuffle_labels" set shuffle_labels=%~2 & shift & shift & goto parse_args
if "%~1"=="-k" set k=%~2 & shift & shift & goto parse_args
if "%~1"=="-n_iter" set n_iter=%~2 & shift & shift & goto parse_args
if "%~1"=="-n_iter_features" set n_iter_features=%~2 & shift & shift & goto parse_args
if "%~1"=="-feature_sample_ratio" set feature_sample_ratio=%~2 & shift & shift & goto parse_args
goto parse_args

:end_args
:: Verificar parámetros obligatorios
if "%project_name%"=="" (
    echo Error: el parámetro -project_name es obligatorio.
    goto :eof
)

:: Configurar selección de características
if %n_iter_features% == 0 set feature_selection=0

:: Llamar a los scripts de Python
python "C:\Users\CNC Audio\gonza\scripts\shared\train_models.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %k% %n_iter% %n_iter_features% %feature_sample_ratio%
python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bca.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %feature_selection% %k% 
python "C:\Users\CNC Audio\gonza\scripts\shared\test_models.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %k%

:: Mostrar los parámetros usados
echo   Pipeline executed with:
echo   project_name=%project_name%
echo   hyp_opt=%hyp_opt%
echo   filter_outliers=%filter_outliers%
echo   shuffle_labels=%shuffle_labels%
echo   k=%k%
echo   n_iter=%n_iter%
echo   n_iter_features=%n_iter_features%
echo   feature_sample_ratio=%feature_sample_ratio%

pause