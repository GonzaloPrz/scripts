import wave
import contextlib
from pathlib import Path
import pandas as pd
import subprocess

def get_wav_duration(filepath):
    """
    Get the duration of a WAV file without loading the entire file.
    """
    with contextlib.closing(wave.open(str(filepath), 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def create_textgrid(input_txt, output_textgrid, start_time=0.0, end_time=10.0):
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
        with open(input_txt, 'r', encoding='utf-8') as file:
            content = file.read().strip()

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
        print(f"Input text file not found: {input_txt}")
    except Exception as e:
        print(f"Error creating .TextGrid file: {e}")

base_dir = Path(Path.home(),'data','redlat') if '/Users/gp' in str(Path.home()) else Path('D:\\CNC_Audio\\data\\13_redlat\\REDLAT_06-02-25')
transcripts = pd.read_excel(Path(base_dir,"REDLAT_FUGU_transcriptions.xlsx"))

centers = [folder.name for folder in base_dir.iterdir() if folder.is_dir()]

for center in centers:
    ids = [folder.name for folder in Path(base_dir,center).iterdir() if folder.is_dir()]

    for id in ids:
        filename = f'REDLAT_{id}_Fugu.wav'
        try:
            transcript = transcripts[transcripts["filename"] == filename]["transcript"].values[0]
            txt_file = filename.replace('.wav','.txt')
            with open(Path(base_dir,center,id,txt_file), 'w', encoding='utf-8') as f:
                f.write(transcript)
        except:
            print(f"Transcript for {filename} not found.")
            continue
        
        try:
            duration = get_wav_duration(Path(base_dir,center,id,filename))
            Path(base_dir,center,id,'ffmpeg').mkdir(parents=True, exist_ok=True)

            subprocess.run(["ffmpeg", "-i", str(Path(base_dir,center,id,filename)),"-ar", "16000", "-ac", "1", "-sample_fmt", "s16",str(Path(base_dir,center,id,"ffmpeg",filename))], check=True)
            textgrid_file = filename.replace('.wav','.TextGrid')
            create_textgrid(Path(base_dir,center,id,txt_file), Path(base_dir,center,id,'ffmpeg',textgrid_file), end_time=duration)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue