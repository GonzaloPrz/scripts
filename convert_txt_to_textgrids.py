import textgrid
from pathlib import Path
import soundfile as sf

tasks = ["Fugu"]

def save_transcript_to_textgrid(transcript, timestamps, filename="output.TextGrid"):
    """
    Saves a transcript with timestamps into a TextGrid file.
    
    :param transcript: List of transcript texts
    :param timestamps: List of (start_time, end_time) tuples for each transcript
    :param filename: Output filename for the TextGrid file
    """
    tg = textgrid.TextGrid()
    tier = textgrid.IntervalTier(name="Transcription", minTime=0.0, maxTime=timestamps[-1][1])
    
    for (start, end), text in zip(timestamps, transcript):
        tier.add(start, end, text)
    
    tg.append(tier)
    tg.write(filename)
    print(f"TextGrid file saved as {filename}")

path_to_data = Path(f"/Users/gp/data/ad_mci_hc/Audios/wavs")
files = [file for file in path_to_data.glob("*.txt") if Path(path_to_data,file.stem + ".wav").exists() and 'sentence' not in file.stem]
for file in files:
    try:
        with open(file, "r",encoding='utf-16') as f:
            transcript = f.read().strip().split("\n")
            audio = file.stem + ".wav"
            audio, fs = sf.read(path_to_data / audio)
            duration = len(audio) / fs
            #Get timestamps based on audio length
            timestamps = [(0,duration)]    

            save_transcript_to_textgrid(transcript, timestamps, filename=Path(path_to_data,file.stem + ".TextGrid"))
    except:
        print(f"Error processing {file}")
        continue