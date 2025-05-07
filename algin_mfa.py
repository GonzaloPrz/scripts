from pathlib import Path
import subprocess
import os


# Define paths
base_dir = r"/Users/gp/data/ad_mci_hc/Audios/wavs/diarize"
# Define model names
dictionary = "spanish_mfa"
acoustic_model = "spanish_mfa"

output_path = str(Path(base_dir,"mfa_aligned_diarize"))
Path(output_path).mkdir(parents=True, exist_ok=True)
# Build the MFA command
command = [
    "mfa",
    "align",
    "--clean",  # Optional: cleans up previous runs
    str(base_dir),
    dictionary,
    acoustic_model,
    output_path
]

# Run the command
try:
    subprocess.run(command, check=True)
    print("Alignment completed successfully.")
except subprocess.CalledProcessError as e:
    print("Error during alignment:")
    print(e)