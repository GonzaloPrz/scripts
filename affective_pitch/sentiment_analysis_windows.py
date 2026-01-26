import pandas as pd
from pathlib import Path
from pysentimiento import create_analyzer
import spacy
import numpy as np

base_dir = Path(Path.home(),'data','affective_pitch') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')
transcripts = pd.read_csv(Path(base_dir,'transcripts_fugu_matched_group.csv'))
analyzer = create_analyzer(task='sentiment',lang="es")

for r, row in transcripts.iterrows():

    transcript = row['transcript']
    if str(transcript) == 'nan':
        continue
    
    transcript = transcript.replace('...','.').replace('..','.')
    
    #Split transcripts into sliding windows of 10 words with an overlap of 5 words
    
    sentences = []
    words = transcript.split(' ')
    window_size = 10
    for i in range(window_size//2, len(words) - window_size//2, window_size//2):
        window_words = words[np.max((i-window_size//2,0)):np.min((i + window_size//2,len(words)))]
        sentence = ' '.join(window_words)
        sentences.append(sentence)
    
    if len(sentences) == 0:
        continue
    
    sentiments = np.full(len(sentences),fill_value=np.nan,dtype=object)
    probas_pos = np.full(len(sentences),fill_value=np.nan,dtype=object)
    probas_neg = np.full(len(sentences),fill_value=np.nan,dtype=object)
    probas_neu = np.full(len(sentences),fill_value=np.nan,dtype=object)

    for s,sentence in enumerate(sentences):

        sentiment = analyzer.predict(sentence)
        sentiments[s] = sentiment.output
        probas_pos[s] = sentiment.probas['POS']
        probas_neg[s] = sentiment.probas['NEG']
        probas_neu[s] = sentiment.probas['NEU']
    
    #Eliminate nan values
    sentiments = np.array([sentiments[i] for i in range(len(sentiments)) if not pd.isna(sentiments[i])])
    probas_pos = np.array([probas_pos[i] for i in range(len(probas_pos)) if not pd.isna(probas_pos[i])])
    probas_neg = np.array([probas_neg[i] for i in range(len(probas_neg)) if not pd.isna(probas_neg[i])])
    probas_neu = np.array([probas_neu[i] for i in range(len(probas_neu)) if not pd.isna(probas_neu[i])])

    transcripts.at[r,'sentiments'] = str(sentiments)
    transcripts.at[r,'pos_proba'] = str(probas_pos)
    transcripts.at[r,'neg_proba'] = str(probas_neg)
    transcripts.at[r,'neu_proba'] = str(probas_neu)
    transcripts.at[r,'pos_proba_norm'] = str(probas_pos/sum(probas_pos))
    transcripts.at[r,'neg_proba_norm'] = str(probas_neg/sum(probas_neg))
    transcripts.at[r,'neu_proba_norm'] = str(probas_neu/sum(probas_neu))

    transcripts.at[r,'Fugu__sentiment__ratio_POS'] = sum([1 for sent in sentiments if sent == 'POS']) / len(sentences)
    transcripts.at[r,'Fugu__sentiment__ratio_NEG'] = sum([1 for sent in sentiments if sent == 'NEG']) / len(sentences)
    transcripts.at[r,'Fugu__sentiment__ratio_NEU'] = sum([1 for sent in sentiments if sent == 'NEU']) / len(sentences)

transcripts.to_csv(Path(base_dir,'transcripts_fugu_matched_group_sentiment_windows.csv'),index=False)