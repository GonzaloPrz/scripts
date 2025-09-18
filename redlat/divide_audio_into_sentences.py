from textgrid import TextGrid
from pathlib import Path
import scipy.io.wavfile as wave
import docx2txt

def parse_word_alignments(textgrid_file):
    tg = TextGrid.fromFile(textgrid_file)
    word_tier = tg.getFirst('words')  # Get the "words" tier

    word_alignments = []
    for interval in word_tier.intervals:
        if interval.mark.strip():  # Ignore empty intervals
            word_alignments.append({
                "word": interval.mark.strip(),
                "start": interval.minTime,
                "end": interval.maxTime
            })
    return word_alignments

def get_sentence_boundaries(word_alignments, sentences):
    sentence_boundaries = []

    current_word_index = 0
    for sentence in sentences:
        
        words = [word.replace(',','').replace(';','').replace('.','').lower() for word in sentence.split(' ') if word not in [',',',',':',';','"','(',')',' ']]

        if len(words) == 0:
            continue
        try:
            sentence_start = word_alignments[current_word_index]["start"]
            sentence_end = word_alignments[current_word_index + len(words) - 1]["end"]
        except:
            continue
        current_word_index += len(words)
        
        sentence_boundaries.append({
            "sentence": sentence,
            "start": sentence_start,
            "end": sentence_end
        })

    return sentence_boundaries

def split_audio(audio_file, sentence_boundaries, output_dir):
    sr, audio = wave.read(audio_file)
    #Get the sample rate of the audio
    for i, boundary in enumerate(sentence_boundaries):
        start = boundary["start"]
        end = boundary["end"]
        sentence = boundary["sentence"]
        output_path = f"{output_dir}/{audio_file.stem}_sentence_{i + 1}.wav"
        #Write to .wav
        audio_segment = audio[int(start * sr):int(end * sr)]
        wave.write(output_path, sr, audio_segment)
        print(f"Exported: {output_path}")
        with open(output_path.replace(".wav",".txt"), 'w') as f:
            f.write(sentence.strip())

base_dir = Path("/Users/gp/data/redlat") if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','redlat')

Path(base_dir,'sentences').mkdir(exist_ok=True)

for textgrid_file in Path(base_dir,"mfa_aligned").glob('*.TextGrid'):
    audio_file_list = list(base_dir.rglob(f'{textgrid_file.stem}.wav'))
    if len(audio_file_list) == 0:
        continue
    else:
        audio_file = audio_file_list[0]
    
    if not audio_file.exists():
        continue
    output_dir = Path(base_dir,'sentences')
    output_dir.mkdir(exist_ok=True)
    txt_file = str(audio_file).replace('_diarize.wav','_mono_16khz_diarize_loudnorm_denoised.txt')

    try:
        with open(txt_file, 'r',encoding='utf-8') as f:
            original_sentences = f.read().replace("...","").replace('¿','').replace('?','.').split('.')
    except:

        if not Path(txt_file.replace('.txt','.docx')).exists():
            print(f"Error processing file {audio_file}")
            continue
        original_sentences = docx2txt.process(Path(txt_file.replace('.txt','.docx'))).replace("...","").replace('¿','').replace('?','.').split('. ')

    word_alignments = parse_word_alignments(textgrid_file)

    sentence_boundaries = get_sentence_boundaries(word_alignments, original_sentences)
    
    split_audio(audio_file, sentence_boundaries, output_dir)
