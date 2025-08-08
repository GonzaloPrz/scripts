from pathlib import Path
import subprocess
import pandas as pd
import os
import shutil

# --- Setup Environment & Paths ---

# Set a dedicated temporary directory for MFA to use.
# This helps avoid conflicts with MFA's default location (in C:\Users\...)
env = os.environ.copy()
mfa_temp_dir = Path("D:/mfa_output")
mfa_temp_dir.mkdir(exist_ok=True)
env["MFA_OUTPUT_DIRECTORY"] = str(mfa_temp_dir)

# Define the base directory for your project data.
# This checks if you're on a Mac or Windows and sets the path accordingly.
base_dir = Path(Path.home(),'data','redlat') if '/Users/gp' in str(Path.home()) else Path('D:\\CNC_Audio\\data\\13_redlat\\REDLAT_06-02-25')

# Load the transcriptions from the Excel file.
try:
    transcripts_df = pd.read_excel(Path(base_dir,"REDLAT_FUGU_transcriptions.xlsx"))
except FileNotFoundError:
    print(f"Error: The transcript file was not found at {Path(base_dir,'REDLAT_FUGU_transcriptions.xlsx')}")
    exit()

# Define model names for clarity.
acoustic_model = "spanish_mfa"
dictionary = "spanish_latin_america_mfa"

# Define and create the final output directory for the aligned TextGrids.
output_path = Path(base_dir, "mfa_aligned")
output_path.mkdir(parents=True, exist_ok=True)

# --- Main Processing Loop ---

# Get the list of center directories (e.g., 'Avila').
centers = [folder.name for folder in base_dir.iterdir() if folder.is_dir() and folder.name != 'mfa_aligned']

print(f"Found {len(centers)} centers to process...")

for center in centers:
    center_path = Path(base_dir, center)
    # Get the list of ID directories (e.g., 'AF051').
    ids = [folder.name for folder in center_path.iterdir() if folder.is_dir() and folder.name != 'mfa_aligned']

    for id_code in ids:
        print(f"\n--- Processing {center}/{id_code} ---")
        
        # This is the directory containing the source .wav and where we'll create the .lab file.
        corpus_dir = Path(center_path, id_code, "ffmpeg")
        if not corpus_dir.exists():
            print(f"Warning: Corpus directory not found, skipping: {corpus_dir}")
            continue

        audio_filename = f'REDLAT_{id_code}_Fugu.wav'
        lab_filename = f'REDLAT_{id_code}_Fugu.lab'
        lab_filepath = corpus_dir / lab_filename

        # --- FIX #1: Find transcript and write it to a .lab file for MFA ---
        try:
            # Look up the transcript in the DataFrame.
            transcript_series = transcripts_df[transcripts_df["filename"] == audio_filename]["transcript"]
            if transcript_series.empty:
                print(f"Transcript for {audio_filename} not found in Excel. Skipping.")
                continue
            transcript = transcript_series.values[0]
            
            # Write the transcript to the .lab file in UTF-8.
            with open(lab_filepath, "w", encoding="utf-8") as f:
                f.write(str(transcript))
            print(f"Created transcript file: {lab_filepath}")

        except Exception as e:
            print(f"An error occurred while preparing the transcript for {audio_filename}: {e}")
            continue
        
        # --- NEW FIX: Manually remove old MFA cache to prevent lock errors ---
        # MFA creates a '.mfa' directory inside the corpus_dir. If a previous run
        # crashed, files inside can remain locked. We remove it first.
        mfa_cache_dir = corpus_dir / '.mfa'
        if mfa_cache_dir.is_dir():
            print(f"Found old MFA cache. Removing: {mfa_cache_dir}")
            try:
                shutil.rmtree(mfa_cache_dir)
            except OSError as e:
                print(f"Error removing cache directory: {e}. This could be a permissions issue or the file is still in use. Skipping this item.")
                continue

        # --- FIX #2: Build the MFA command with corrected argument order ---
        command = [
            r"C:\mfa_env\python.exe",
            "-m", "montreal_forced_aligner",
            "align",
            # --- Positional arguments must come before optional flags ---
            str(corpus_dir),        # 1. Path to the data (contains .wav and .lab)
            dictionary,             # 2. Dictionary name/path
            acoustic_model,         # 3. Acoustic model name/path
            str(output_path),       # 4. Output directory for TextGrids
            # --- Optional flags come after positional arguments ---
            "--clean",
            "--beam", "100",
            "--retry_beam", "400",
        ]

        # --- Run the MFA command ---
        try:
            print(f"Running MFA for {audio_filename}...")
            result = subprocess.run(
                            command,
                            check=True,       # Raises an error if the command fails
                            capture_output=True,
                            text=True,
                            env=env
                        )
            print("Alignment completed successfully.")
            # print(result.stdout) # Uncomment for verbose success output
        except subprocess.CalledProcessError as e:
            print("Error during alignment:")
            # The stderr from MFA usually has the most useful error message.
            print(f"STDERR:\n{e.stderr}")
            print(f"STDOUT:\n{e.stdout}")


print("\n--- All processing finished. ---")