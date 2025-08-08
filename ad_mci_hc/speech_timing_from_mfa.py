from pathlib import Path
import pandas as pd
import textgrid 
import pyphen
import numpy as np

def contar_silabas(palabras, idioma='es_ES'):
    diccionario = pyphen.Pyphen(lang=idioma)
    resultados = []

    for palabra in palabras:
        silabeada = diccionario.inserted(palabra)
        cantidad = silabeada.count('-') + 1 if silabeada else 1
        resultados.append(cantidad)

    return resultados

def get_speech_timing_from_mfa(mfa_path: Path) -> pd.DataFrame:
    """
    Extracts speech timing information from a Montreal Forced Aligner (MFA) TextGrid file.

    Args:
        mfa_path (Path): Path to the MFA TextGrid file.
        output_path (Path): Path to save the output CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the speech timing information.
    """
    # Load the TextGrid file
    tg = textgrid.TextGrid.fromFile(mfa_path)

    # Extract the intervals from the first tier
    intervals = tg[0].intervals
    #Extract information from intervals

    df = pd.DataFrame({
        'start_time': [interval.minTime for interval in intervals],
        'end_time': [interval.maxTime for interval in intervals],
        'label': [interval.mark for interval in intervals]
    })

    pauses = df[df['label'] == ""]
    pauses = pauses[pauses['end_time'] - pauses['start_time'] > 0.3]

    phonation = df[df['label'] != ""]

    pause_time = 0
    phonation_time = 0
    for i in range(len(pauses)):
        pause_time += (pauses.iloc[i]['end_time'] - pauses.iloc[i]['start_time'])
    for i in range(len(phonation)):
        phonation_time += (phonation.iloc[i]['end_time'] - phonation.iloc[i]['start_time'])

    speech_time = phonation_time + pause_time

    syll = contar_silabas(phonation['label'].tolist())
    nsyll_mean = np.mean(syll)
    nsyll_std = np.std(syll)
    speechrate = len(syll) / speech_time if speech_time > 0 else 0
    articulation_rate = len(syll) / phonation_time if phonation_time > 0 else 0
    mean_syllable_duration = phonation_time / len(syll) if len(syll) > 0 else 0

    df = pd.DataFrame({
        'phonation_time': [phonation_time],
        'pause_time': [pause_time],
        'fugu__talking_intervals_mfa__pause_duration_mean': [pause_time / len(pauses) if len(pauses) > 0 else 0],
        'fugu__talking_intervals_mfa__pause_duration_std': [pauses['end_time'].std() if len(pauses) > 0 else 0],
        'fugu__talking_intervals_mfa__pause_duration_min': [pauses['end_time'].min() if len(pauses) > 0 else 0],
        'fugu__talking_intervals_mfa__pause_duration_max': [pauses['end_time'].max() if len(pauses) > 0 else 0],
        'fugu__talking_intervals_mfa__num_pauses': [len(pauses)],
        'fugu__talking_intervals_mfa__nsyll_mean': [nsyll_mean],
        'fugu__talking_intervals_mfa__nsyll_std': [nsyll_std],
        'fugu__talking_intervals_mfa__speech_rate': [speechrate],
        'fugu__talking_intervals_mfa__articulation_rate': [articulation_rate],
        'fugu__talking_intervals_mfa__mean_syllable_duration': [mean_syllable_duration],
        'total_time': [phonation_time + pause_time]
    })

    return df

path_to_data = r"/Users/gp/data/ad_mci_hc/Audios/wavs/diarize/mfa_aligned_diarize"

data = pd.DataFrame()
for file in Path(path_to_data).glob("*.TextGrid"):
    print(file)
    df = get_speech_timing_from_mfa(file)
    df["id"] = file.stem.split("__")[0].replace("T1_","")
    if data.empty:
        data = df
    else:
        data = pd.concat([data, df], ignore_index=True)

data.to_csv(Path(Path(path_to_data).parent.parent.parent.parent,r"speech_timing_mfa.csv"), index=False)    