@echo off
:: Activate virtual environment
call "C:\Users\CNC Audio\gonza\gonza-env\Scripts\activate"

::Execute Python scripts with arguments passed directly
python "C:\Users\CNC Audio\gonza\scripts\shared\train_models_bayes.py" %*
python "C:\Users\CNC Audio\gonza\scripts\shared\bootstrap_models_bayes.py"
python "C:\Users\CNC Audio\gonza\scripts\shared\train_final_model_bayes.py"
python "C:\Users\CNC Audio\gonza\scripts\shared\test_final_model_bayes.py"
:: Display used parameters for logging
echo Pipeline executed with arguments: %*