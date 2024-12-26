@echo off
echo train_models...
python "C:\Users\CNC Audio\gonza\scripts\shared\train_models.py"

echo test_models...
python "C:\Users\CNC Audio\gonza\scripts\shared\test_models.py"

echo bootstrap_models...
python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bca.py"

echo report_best_models...
python "C:\Users\CNC Audio\gonza\scripts\shared\report_best_models.py"

echo plot_predictions...
python "C:\Users\CNC Audio\gonza\scripts\shared\plot_predictions.py"

echo All scripts completed successfully.