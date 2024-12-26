@echo off
echo Running first script...
start python "C:\Users\CNC Audio\gonza\scripts\shared\test_models.py"

echo Running second script...
start python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bca.py"

echo Running third script...
start python "C:\Users\CNC Audio\gonza\scripts\shared\report_best_models.py"

echo All scripts completed successfully.