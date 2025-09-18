from pathlib import Path
import subprocess
import pandas as pd
import os
import shutil
import docx2txt

# --- Setup Environment & Paths ---

# Set a dedicated temporary directory for MFA to use.
# This helps avoid conflicts with MFA's default location (in C:\Users\...)
env = os.environ.copy()
mfa_temp_dir = Path("D:/mfa_output") if Path('D:').exists() else Path('/Users/gp/mfa_output')
mfa_temp_dir.mkdir(exist_ok=True)

env["MFA_OUTPUT_DIRECTORY"] = str(mfa_temp_dir)

# Define the base directory for your project data.
# This checks if you're on a Mac or Windows and sets the path accordingly.
base_dir = Path(Path.home(),'data','redlat') if '/Users/gp' in str(Path.home()) else Path('D:\\CNC_Audio\\data\\13_redlat\\REDLAT_06-02-25')

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
    evals = [folder.name for folder in center_path.iterdir() if folder.is_dir() and folder.name != 'mfa_aligned']

    for eval_code in evals:
        corpus_dir = Path(center_path, eval_code)

        files = [file.name for file in Path(base_dir,center,eval_code).iterdir() if file.suffix == '.wav']

        print(f"\n--- Processing {center}/{eval_code} ---")
        
        # This is the directory containing the source .wav and where we'll create the .lab file.
        if not corpus_dir.exists():
            print(f"Warning: Corpus directory not found, skipping: {corpus_dir}")
            continue
        for filename in files:
            lab_filename = filename.replace('.wav','.lab')
            lab_filepath = corpus_dir / lab_filename

            # --- FIX #1: Find transcript and write it to a .lab file for MFA ---
            try:
                # Look up the transcript in the DataFrame.
                txt_filename = filename.replace('_diarize.wav','_mono_16khz_diarize_loudnorm_denoised.txt')

                try:
                    with open(Path(base_dir,center,eval_code,txt_filename), 'r',encoding='utf-16') as f:
                        transcript = f.read()
                except:
                    if not Path(base_dir,center,eval_code,txt_filename.replace('.txt','.docx')).exists():
                        print(f"Error processing file {filename}")
                        continue
                    transcript = docx2txt.process(Path(base_dir,center,eval_code,txt_filename.replace('.txt','.docx')))

                # Write the transcript to the .lab file in UTF-8.
                with open(lab_filepath, "w", encoding="utf-8") as f:
                    f.write(str(transcript))
                print(f"Created transcript file: {lab_filepath}")

            except Exception as e:
                print(f"An error occurred while preparing the transcript for {filename}: {e}")
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
            mfa_path = r'C:\mfa_env\python.exe' if '/Users/gp' not in str(Path.home()) else '/opt/anaconda3/envs/mfa/bin/python'
            command = [
                mfa_path,
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
                print(f"Running MFA for {filename}...")
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