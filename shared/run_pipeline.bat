@echo off
echo bootstrap_models...
python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bca.py"

echo test_models...
python "C:\Users\CNC Audio\gonza\scripts\shared\test_models.py"

echo report_best_models...
python "C:\Users\CNC Audio\gonza\scripts\shared\report_best_models.py"

echo All scripts completed successfully.