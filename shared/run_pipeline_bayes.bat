@echo off

:: Inicializar variables con valores por defecto
set project_name=""
set feature_selection=1
set filter_outliers=1
set shuffle_labels=0
set k=5
set n_iter=15
set init_points=20

:: Procesar argumentos
:parse_args
if "%~1"=="" goto end_args
if "%~1"=="-project_name" set project_name=%~2 & shift & shift & goto parse_args
if "%~1"=="-feature_selection" set feature_selection=%~2 & shift & shift & goto parse_args
if "%~1"=="-filter_outliers" set filter_outliers=%~2 & shift & shift & goto parse_args
if "%~1"=="-shuffle_labels" set shuffle_labels=%~2 & shift & shift & goto parse_args
if "%~1"=="-k" set k=%~2 & shift & shift & goto parse_args
if "%~1"=="-n_iter" set n_iter=%~2 & shift & shift & goto parse_args
if "%~1"=="-init_points" set n_iter_features=%~2 & shift & shift & goto parse_args
goto parse_args

:end_args
:: Verificar parámetros obligatorios
if "%project_name%"=="" (
    echo Error: el parámetro -project_name es obligatorio.
    goto :eof
)

:: Configurar selección de características
if %n_iter_features% == 0 set feature_selection=0

call "C:\Users\CNC Audop\gonza\gonza-env\Scripts\activate"

:: Llamar a los scripts de Python
python "C:\Users\CNC Audio\gonza\scripts\shared\train_models_bayes.py" %project_name% %feature_selection% %filter_outliers% %shuffle_labels% %k% %n_iter% %init_points%
python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bayes.py" %project_name% %feature_selection% %filter_outliers% %shuffle_labels% %k% 

:: Mostrar los parámetros usados
echo   Pipeline executed with:
echo   project_name=%project_name%
echo   feature_selection=%feature_selection%
echo   filter_outliers=%filter_outliers%
echo   shuffle_labels=%shuffle_labels%
echo   k=%k%
echo   n_iter=%n_iter%
echo   init_points=%n_iter_features%