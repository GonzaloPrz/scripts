from pathlib import Path
import sys,tqdm

sys.path.append(str(Path(Path.home(),'local_feature_extraction','audio_features','preprocessing')))

project_name = 'GeroApathy'

from audio_preprocess_pipeline import preprocess_audio

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

tasks = ['DiaTipico']

for task in tasks:
    path_to_audios = Path(data_dir,f'{task}_audios')
    for audio in tqdm.tqdm(path_to_audios.rglob('*.wav')):
        print(f'Preprocessing {audio.stem}')

        if Path(path_to_audios.parent,'diarize',f'{audio.stem}_mono_16khz_loudnorm_denoised.wav').exists():
            continue

        preprocess_audio(audio,str(path_to_audios),
                        diarize_config=False,
                        vad_config=True)