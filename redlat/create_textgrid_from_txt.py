import wave
import contextlib
from pathlib import Path
import pandas as pd
import subprocess
import docx2txt

def get_wav_duration(filepath):
    """
    Get the duration of a WAV file without loading the entire file.
    """
    with contextlib.closing(wave.open(str(filepath), 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def create_textgrid(content, output_textgrid, start_time=0.0, end_time=10.0):
    """
    Converts a .txt file to a .TextGrid file with a single interval tier.
    
    Parameters:
        input_txt (str): Path to the input .txt file.
        output_textgrid (str): Path to the output .TextGrid file.
        start_time (float): Start time for the TextGrid intervals.
        end_time (float): End time for the TextGrid intervals.
    """
    try:
        # Read the text content from the .txt file
        
        # Escape double quotes in content
        content = content.replace('"', '""')

        # Create the TextGrid structure
        textgrid_content = f'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = {start_time}
xmax = {end_time}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = {start_time}
        xmax = {end_time}
        intervals: size = 1
        intervals [1]:
            xmin = {start_time}
            xmax = {end_time}
            text = "{content}"
'''

        # Write the TextGrid content to the output file
        with open(output_textgrid, 'w', encoding='utf-8') as file:
            file.write(textgrid_content)

        print(f".TextGrid file created: {output_textgrid}")

    except FileNotFoundError:
        print(f"Input text file not found")
    except Exception as e:
        print(f"Error creating .TextGrid file: {e}")

base_dir = Path(Path.home(),'data','redlat') if '/Users/gp' in str(Path.home()) else Path('D:\\CNC_Audio\\data\\13_redlat\\REDLAT_06-02-25')

centers = [folder.name for folder in base_dir.iterdir() if folder.is_dir()]

for center in centers:
    evals = [folder.name for folder in Path(base_dir,center).iterdir() if folder.is_dir()]

    for eval_code in evals:
        files = [file.name for file in Path(base_dir,center,eval_code).iterdir() if file.suffix == '.wav']

        for filename in files:
            txt_filename = filename.replace('_diarize.wav','_mono_16khz_diarize_loudnorm_denoised.txt')

            try:
                with open(Path(base_dir,center,eval_code,txt_filename), 'r',encoding='utf-16') as f:
                    transcript = f.read()
            except:
                if not Path(base_dir,center,eval_code,txt_filename.replace('.txt','.docx')).exists():
                    print(f"Error processing file {filename}")
                    continue
                
                transcript = docx2txt.process(Path(base_dir,center,eval_code,txt_filename.replace('.txt','.docx')))

            try:
                duration = get_wav_duration(Path(base_dir,center,eval_code,filename))
                Path(base_dir,center,eval_code).mkdir(parents=True, exist_ok=True)

                #subprocess.run(["ffmpeg", "-i", str(Path(base_dir,center,eval_code,filename)),"-ar", "16000", "-ac", "1", "-sample_fmt", "s16",str(Path(base_dir,center,eval_code,"ffmpeg",filename))], check=True)
                textgrid_file = filename.replace('.wav','.TextGrid')
                create_textgrid(transcript, Path(base_dir,center,eval_code,textgrid_file), end_time=duration)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue