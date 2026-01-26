import pandas as pd
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import shutil

base_dir = Path(Path.home(),'data','affective_pitch') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')
audio_dir = Path('D:','CNC_Audio','data','13_redlat','REDLAT_24-09-25_masked_prepro')
output_dir = Path(base_dir,'divided_audios','windows')
Path(output_dir,'POS').mkdir(parents=True, exist_ok=True)
Path(output_dir,'NEG').mkdir(parents=True, exist_ok=True)
Path(output_dir,'NEU').mkdir(parents=True, exist_ok=True)
Path(output_dir, 'ALL').mkdir(parents=True, exist_ok=True)

sites = [folder.name for folder in audio_dir.iterdir() if folder.is_dir()]
transcripts = pd.read_csv(Path(base_dir,'transcripts_fugu_matched_group_sentiment_windows.csv'))

for r, row in transcripts.iterrows():
    audio_path = Path(audio_dir,row['site'],row['id'],f'REDLAT_{row["id"]}_Fugu.wav')
    #Move audio to output_dir

    try:
        Path(output_dir.parent.parent.parent / 'all_audios').mkdir(exist_ok=True)
        shutil.copy2(audio_path,Path(output_dir.parent.parent.parent,'all_audios',audio_path.name))
    except:
        print(f'Error copying file {audio_path.name}')
    '''
    transcript = row['transcript']

    if str(transcript) == 'nan':
        continue

    transcript = transcript.replace('...','.').replace('..','.')

    timestamps = row['word_timestamps'].split('],')
    timestamps = [eval(item.replace('[','').replace(']','')) for item in timestamps]
    timestamps = [{'word': item[0].replace('...','.').replace('..','.'), 'start_time': item[1], 'end_time': item[2]} for item in timestamps]
    if isinstance(row['sentiments'],float):
        continue

    sentiments = row['sentiments'].replace('[','').replace(']','').split(' ')
    sentiments = [s.strip().replace("'",'') for s in sentiments]
        
    sentences = []
    words = transcript.split(' ')
    window_size = 10
    for i in range(window_size//2, len(words) - window_size//2, window_size//2):
        window_words = words[np.max((i-window_size//2,0)):np.min((i + window_size//2,len(words)))]

        sentence = ' '.join(window_words)
        sentences.append(sentence)
    
    try:
        audio, sr = sf.read(audio_path)
    except:
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            continue

        with open(audio_path,'rb') as f:
            audio = AudioSegment.from_file(f)
            sr = audio.frame_rate
            audio.export(audio_path, format="wav") 
        audio, sr = sf.read(audio_path)
    s = 0
    for sentence in sentences:
        if len(sentence) == 0:
            continue

        sentence_words = sentence.split()
        start_time = None
        end_time = None

        for i,word in enumerate(sentence_words):
            try:
                word_info = timestamps[s:s+window_size][i]
            except IndexError:
                word_info = None
                print(timestamps)
            if word_info:
                if start_time is None:
                    start_time = word_info['start_time']
                end_time = word_info['end_time']

        if start_time is not None and end_time is not None:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            sentence_audio = audio[start_sample:end_sample]

            sentence_filename = f"{row['id']}_Fugu__window_{s+1}.wav"
            sentence_output_path = Path(output_dir, sentiments[s],sentence_filename)
            sf.write(sentence_output_path, sentence_audio, sr)
            sentence_output_path = Path(output_dir, 'ALL',sentence_filename)
            sf.write(sentence_output_path, sentence_audio, sr)
            
            s += 1
        else:
            print(f"Could not find timestamps for sentence: {sentence} in transcript ID: {row['id']}")
        '''