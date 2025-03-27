@echo off
:: Activate virtual environment
::call "C:\Users\CNC Audio\gonza\gonza-env\Scripts\activate"

:: Execute Python scripts with arguments passed directly
python "C:\Users\CNC Audio\gonza\scripts\shared\train_models.py" %*
python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bca.py"
python "C:\Users\CNC Audio\gonza\scripts\shared\test_models.py"
python "C:\Users\CNC Audio\gonza\scripts\shared\report_best_models.py"
python "C:\Users\CNC Audio\gonza\scripts\shared\train_final_model.py"
python "C:\Users\CNC Audio\gonza\scripts\shared\plot_scores.py"
:: Display used parameters for logging
echo Pipeline executed with arguments: %*
