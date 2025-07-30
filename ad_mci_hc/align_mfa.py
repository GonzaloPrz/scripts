# filepath: c:\Users\gperez\scripts\ad_mci_hc\align_mfa.py
from pathlib import Path
import subprocess

# Define paths
base_dir = Path(Path.home(),'data','redlat') if '/Users/gp' in str(Path.home()) else Path('D:\\CNC_Audio\\data\\redlat')

# Define model names
dictionary = "spanish_mfa"
acoustic_model = "spanish_latin_america_mfa"

output_path = Path(base_dir, "mfa_aligned")

# Ensure directories exist
output_path.mkdir(parents=True, exist_ok=True)

# Build the MFA command
command = [
    r"C:\mfa_env\Scripts\mfa.exe",
    "align",
    "--clean",
    str(base_dir),
    "spanish_latin_america_mfa",
    "spanish_mfa",
    str(output_path)
]

# Run the command
try:
    subprocess.run(command, check=True)
    print("Alignment completed successfully.")
except subprocess.CalledProcessError as e:
    print("Error during alignment:")
    print(e)