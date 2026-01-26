import pandas as pd
from pathlib import Path
import soundfile as sf
import numpy as np
import torch

import torchaudio

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

def read_audio(path, target_sr=16000):
    x, sr = sf.read(str(path), always_2d=True)   # (n, ch)
    x = x.mean(axis=1)                           # mono
    x = x.astype(np.float32)

    wav = torch.from_numpy(x)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav
path_to_audios = (
    Path(Path.home(), "data", "affective_pitch", "all_audios")
    if "Users/gp" in str(Path.home())
    else Path("D:", "CNC_Audio", "gonza", "data", "affective_pitch", "all_audios")
)

Path(path_to_audios,'speech_segments').mkdir(parents=True, exist_ok=True)

model = load_silero_vad()

rows = []
audios = [p for p in path_to_audios.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]

for audio in audios:
    wav = read_audio(str(audio))  # pass str to be safe
    speech_timestamps = get_speech_timestamps(wav, model, min_speech_duration_ms = 800, min_silence_duration_ms = 1400, max_speech_duration_s = 20, return_seconds=True)
    
    for i, ts in enumerate(speech_timestamps):
        ts['start'] = round(ts['start'], 3)
        ts['end'] = round(ts['end'], 3)

        segment = wav[int(ts['start']*16000):int(ts['end']*16000)]
        segment_path = Path(path_to_audios,'speech_segments',audio.stem + f"_phrase_{i+1}.wav")
        sf.write(segment_path, segment.numpy(), 16000)

    rows.append({"audio_path": str(audio.name), "timestamps": speech_timestamps,
                "length": len(wav) / 16000, "average_utterance_duration": np.mean([ts['end'] - ts['start'] for ts in speech_timestamps]) if speech_timestamps else 0,
                "average_silence_duration": np.mean([speech_timestamps[i]['start'] - speech_timestamps[i-1]['end'] for i in range(1, len(speech_timestamps))]) if len(speech_timestamps) > 1 else 0
               })
    
speech_segments = pd.DataFrame(rows)
speech_segments = speech_segments.explode('timestamps').reset_index(drop=True)
speech_segments.dropna(subset=['timestamps'], inplace=True)
speech_segments[['start', 'end']] = pd.DataFrame(speech_segments['timestamps'].tolist(), index=speech_segments.index)
speech_segments = speech_segments.drop(columns=['timestamps'])

speech_segments.to_csv(Path(path_to_audios.parent,'speech_timestamps.csv'))