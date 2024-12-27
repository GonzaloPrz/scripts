@echo off

:: Check if the user provided the required project_name argument

:: Required parameter
set project_name=%1

:: Optional parameters with default values
set hyp_opt=1
set filter_outliers=1
set shuffle_labels=0
set k=5
set n_iter=50
set n_iter_features=50
set feature_sample_ratio=0.5

if NOT "%2"=="" set hyp_opt=%2
if NOT "%3"=="" set filter_outliers=%3
if NOT "%4"=="" set shuffle_labels=%4
if NOT "%5"=="" set k=%5
if NOT "%6"=="" set n_iter=%6
if NOT "%7"=="" set n_iter_features=%7
if NOT "%8"=="" set feature_sample_ratio=%8

if %n_iter_features% == 0 set feature_selection = 0

:: Call your Python scripts and pass all parameters
python "C:\Users\CNC Audio\gonza\scripts\shared\train_models.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %k% %n_iter% %n_iter_features% %feature_sample_ratio%
python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bca.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %feature_selection% %k% 
python "C:\Users\CNC Audio\gonza\scripts\test_models.py" %project_name% %hyp_opt% %filter_outliers% %shuffle_labels% %k%

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