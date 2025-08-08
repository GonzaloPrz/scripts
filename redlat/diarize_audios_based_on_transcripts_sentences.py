import soundfile as sf
from pathlib import Path
import numpy as np
import pandas as pd
import re
import contextlib,wave

def get_wav_duration(filepath):
    """
    Get the duration of a WAV file without loading the entire file.
    """
    with contextlib.closing(wave.open(str(filepath), 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration
    
base_dir = Path(r"D:\CNC_Audio\data\13_redlat\REDLAT_06-02-25")
transcripts = pd.read_excel(Path(base_dir,"REDLAT_FUGU_transcriptions.xlsx"))
Path(base_dir,"diarize").mkdir(exist_ok=True)

centers = [folder.name for folder in base_dir.iterdir() if folder.is_dir()]

for center in centers:
    ids = [folder.name for folder in Path(base_dir,center).iterdir() if folder.is_dir()]

    for id in ids:
        filename = f'REDLAT_{id}_Fugu.wav'

        diarized_audio = np.empty(0)
        participant_text = ""
        sr = 16000

        transcript = transcripts[transcripts["filename"] == filename]["transcript"].values[0]
        #identify every piece of text that starts and ends with EEE
        evaluator_sentences = re.findall(r'EEE(.*?)EEE', transcript)
        if len(evaluator_sentences) != 0:    
            evaluator_sentences = [sentence.replace("...","").split(". ") for sentence in evaluator_sentences]
            evaluator_sentences = [sentence.replace(".","") for sentence in evaluator_sentences[0]]
        with Path(path_to_sentences,f'{file.stem}.txt').open('r') as f:
            sentence = f.read().replace(".","")
        if (len(evaluator_sentences) > 0) and (sentence in evaluator_sentences):
            continue

        audio, sr = sf.read(file)
        diarized_audio = np.concatenate((diarized_audio,audio))
        with Path(path_to_sentences,f'{file.stem}.txt').open('r') as f:
            sentence = f.read().replace(".","")
        participant_text += sentence.replace("...","").replace("\n","") + " "

        if len(diarized_audio) > 0:
            sf.write(Path(base_dir,'diarize',re.sub(r'_sentence_\d+','',file.stem) + "_participant.wav"),diarized_audio,sr)
            duration = get_wav_duration(Path(base_dir,'diarize',re.sub(r'_sentence_\d+','',file.stem) + "_participant.wav"))
            
            with Path(base_dir,'diarize',re.sub(r'_sentence_\d+','',file.stem) + "_participant.txt").open('w',encoding="utf-8") as f:
                f.write(participant_text.strip())