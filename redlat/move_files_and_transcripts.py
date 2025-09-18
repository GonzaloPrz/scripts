from pathlib import Path
import shutil

wav_dir = Path('D:','CNC_Audio','data','13_redlat','REDLAT_06-02-25')
dir_destino = Path('D:','CNC_Audio','gonza','data','redlat','fugu')
dir_destino.mkdir(exist_ok=True,parents=True)

txt_dir = Path('D:','CNC_Audio','REDLAT_06-02-25_prepro','clean_transcripts','transcripts')
wav_files = wav_dir.rglob('*Fugu*.wav')
txt_files = txt_dir.rglob('*Fugu*.txt')

for wav_file in wav_files:
    shutil.copy2(wav_file,Path(dir_destino,wav_file.name))

for txt_file in txt_files:
    shutil.copy2(txt_file,Path(dir_destino,txt_file.name))
