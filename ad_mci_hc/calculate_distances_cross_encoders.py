from sentence_transformers import SentenceTransformer
import pandas as pd
from utils import *
from pathlib import Path
import numpy as np
import tqdm
from transformers import AutoTokenizer

import re

from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def remove_eee_sentences(text):
    text = text.replace('Eee','eee').replace('Eee','eee').replace('EEE','eee')
    
    texto_limpio = re.sub(r'eee.*?eee', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Limpiar espacios extra resultantes
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    return texto_limpio

data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc')

if not data_dir.exists():
    data_dir = Path('D:','gperez','data','ad_mci_hc')

results_dir = Path(str(data_dir).replace('data','results'))

data_file = Path(data_dir,"transcripts_matched_group_divided.csv")

df = pd.read_csv(data_file, sep=",", encoding="utf-8")

video_descriptions= pd.read_excel(Path(data_dir,'video_description.xlsx'))

distances = pd.DataFrame()

all_max_similarities = []
tokenizer = AutoTokenizer.from_pretrained('hiiamsid/sentence_similarity_spanish_es')

video_sentences = []
for r, row in video_descriptions.iterrows():
    video_sentences += nltk.sent_tokenize(row["description"], language='spanish')

for r, row in tqdm.tqdm(df.iterrows()):

    row["text_dividido"] = remove_eee_sentences(row["text_dividido"])

    sentences = nltk.sent_tokenize(row["text_dividido"], language='spanish')   
    
    max_similarities = np.zeros(len(video_sentences))
    
    for i,video_sentence in enumerate(video_sentences):
        max_similarities[i] = np.nanmax([model.predict([sentence,video_sentence]) for sentence in sentences])
    
    max_similarities_dict = {'id':row['id']}
    max_similarities_dict.update(dict((f'fugu__text_cross_enc__sim_concept_{i}',max_similarities[i]) for i in range(len(video_sentences))))

    if distances.empty:
        distances = pd.DataFrame(max_similarities_dict,index=[0])
    else:
        distances = pd.concat((distances,pd.DataFrame(max_similarities_dict,index=[0])),axis=0)
        
distances.to_csv(Path(data_dir,f'distances_cross_enc.csv'))