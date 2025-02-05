#!/bin/bash

# Activate virtual environment
source "/Users/gp/gonza-env/Scripts/activate"

# Run Python scripts with all passed arguments
python "/Users/gp/scripts/shared/train_models.py" "$@"
python "/Users/gp//scripts/shared/bootstrap_models_bca.py" "$@"
#python "/home/cnc_audio/gonza/scripts/shared/test_models.py" "$@"

# Display used parameters for logging
echo "Pipeline executed with arguments: $@"
