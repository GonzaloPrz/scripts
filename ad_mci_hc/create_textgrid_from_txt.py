import wave
import contextlib
from pathlib import Path

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

base_dir = Path(r"D:\gperez","Audios", "wavs", "diarize")

for wav_file in base_dir.glob('*.wav'):
    duration = get_wav_duration(wav_file)
    txt_file = wav_file.with_suffix('.txt')

    if txt_file.exists():
        textgrid_file = wav_file.with_suffix('.TextGrid')
        create_textgrid(txt_file, textgrid_file, end_time=duration)

    else:
        print(f"Corresponding text file not found for: {wav_file}")
