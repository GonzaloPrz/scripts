from sentence_transformers import SentenceTransformer
import pandas as pd
from utils import *
from pathlib import Path
import ast, pickle
import torch
from transformers import AutoTokenizer, AutoModel

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc')

if not data_dir.exists():
    data_dir = Path('D:','gperez','data','ad_mci_hc')

results_dir = Path(str(data_dir).replace('data','results'))
results_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(Path(data_dir,'video_description.xlsx'))
all_embeddings = []
all_sentences = []

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name) if 'sentence-transformer' not in model_name else None

model = AutoModel.from_pretrained(model_name,trust_remote_code=True) if 'sentence-transformer' not in model_name else SentenceTransformer(model_name)

for r, row in df.iterrows():

    sentences = nltk.sent_tokenize(row["description"], language='spanish')
    all_sentences += sentences

    sentence_embeddings = []

    for sent in sentences:

        try:
            encoded_input = tokenizer(sent, padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                model_output = model(**encoded_input)
            all_embeddings.append(mean_pooling(model_output, encoded_input['attention_mask']))
    
        except:
            all_embeddings.append(model.encode(sent))
            
pickle.dump(all_embeddings,open(Path(results_dir,f'video_embeddings_{model_name.split("/")[1]}.pkl'),'wb'))
pickle.dump(all_sentences,open(Path(results_dir,'video_sentences.pkl'),'wb'))
