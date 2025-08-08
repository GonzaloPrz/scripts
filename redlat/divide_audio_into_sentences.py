from textgrid import TextGrid
from pathlib import Path
import scipy.io.wavfile as wave

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
        
        words = [word for word in sentence.split() if word not in [',',',',':',';','"','(',')']]

        if len(words) == 0:
            continue
        try:
            sentence_start = word_alignments[current_word_index]["start"]
            sentence_end = word_alignments[current_word_index + len(words) - 1]["end"]
        except:
            sentence_start = word_alignments[current_word_index - 1]["start"]
            sentence_end = word_alignments[current_word_index + len(words) - 2]["end"]
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

base_dir = Path("/Users/gp/data/ad_mci_hc","Audios","wavs")
Path(base_dir,'sentences').mkdir(exist_ok=True)

for textgrid_file in Path(base_dir,"mfa_aligned").glob('*.TextGrid'):
    audio_file = Path(base_dir,f'{textgrid_file.stem}.wav')
    if not audio_file.exists():
        continue
    output_dir = Path(base_dir,'sentences')
    output_dir.mkdir(exist_ok=True)
    with open(Path(base_dir,f'{textgrid_file.stem}.txt'), 'r',encoding='utf-16') as f:
        original_sentences = f.read().replace("...","").split('.')
    
    word_alignments = parse_word_alignments(textgrid_file)

    sentence_boundaries = get_sentence_boundaries(word_alignments, original_sentences)
    
    split_audio(audio_file, sentence_boundaries, output_dir)
