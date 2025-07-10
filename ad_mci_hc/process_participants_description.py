from sentence_transformers import SentenceTransformer
import pandas as pd
from utils import *
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name) if 'sentence-transformer' not in model_name else None
model = AutoModel.from_pretrained(model_name,trust_remote_code=True) if 'sentence-transformer' not in model_name else SentenceTransformer(model_name)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc')

if not data_dir.exists():
    data_dir = Path('D:','gperez','data','ad_mci_hc')

results_dir = Path(str(data_dir).replace('data','results'))

data_file = Path(data_dir,"filtered_transcripts_no_hallucinations_matched_group.xlsx")

video_embeddings = pickle.load(open(Path(results_dir,f'video_embeddings_{model_name.split("/")[1]}.pkl'),'rb'))
#Flatten embeddings:

df = pd.read_excel(data_file)
video_sentences= pickle.load(open(Path(results_dir,'video_sentences.pkl'),'rb'))

distances = pd.DataFrame()

all_max_similarities = []
for r, row in df.iterrows():
    sentence_embeddings = []

    sentences = nltk.sent_tokenize(row["text_dividido"], language='spanish')

    sentence_embeddings = []

    for sent in sentences:

        try:
            encoded_input = tokenizer(sent, padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                model_output = model(**encoded_input)

            sentence_embeddings.append(mean_pooling(model_output, encoded_input['attention_mask']))
        except:
            sentence_embeddings.append(model.encode(sent))

    max_similarities = []
    max_similarities_hallucinations = []

    argmax_similarities = []
    argmax_similarities_hallucinations = []

    for video_emb in video_embeddings:
        
        max_similarities.append(np.max([cosine_similarity(np.array(video_emb).reshape(1,-1),np.array(sent_emb).reshape(1,-1)) for sent_emb in sentence_embeddings]))
        argmax_similarities.append(np.argmax([cosine_similarity(np.array(video_emb).reshape(1,-1),np.array(sent_emb).reshape(1,-1)) for sent_emb in sentence_embeddings]))
    
    for sent_emb in sentence_embeddings:
        max_similarities_hallucinations.append(np.max([cosine_similarity(np.array(video_emb).reshape(1,-1),np.array(sent_emb).reshape(1,-1)) for video_emb in video_embeddings]))
        argmax_similarities_hallucinations.append(np.argmax([cosine_similarity(np.array(video_emb).reshape(1,-1),np.array(sent_emb).reshape(1,-1)) for video_emb in video_embeddings]))

    max_similarities_dict = {'id':row['id']}

    max_similarities_dict.update(dict((f'fugu__text__{model_name.split("/")[1]}__sim_concept_{i}',max_similarities[i] if max_similarities[i] > .5 else 0) for i in range(len(max_similarities))))
    #max_similarities_dict.update(dict((f'embedding_concept_{i}',[embeddings_video[i]]) for i in range(len(argmax_similarities))))
    #max_similarities_dict.update(dict((f'embedding_most_similar_sentence_concept_{i}',[sentence_embeddings[argmax_similarities[i]]]) for i in range(len(argmax_similarities))))
    max_similarities_dict.update(dict((f'most_similar_sentence_concept_{i}',f"{video_sentences[i]}: {sentences[argmax_similarities[i]]}") for i in range(len(argmax_similarities))))
    max_similarities_dict.update({f'fugu__text__number_of_omissions':sum(np.array(max_similarities) < .5)/ len(max_similarities)})
    max_similarities_dict.update({f'fugu__text__number_of_hallucinations':sum(np.array(max_similarities_hallucinations) < .5)/len(max_similarities_hallucinations)})
    max_similarities_dict.update({f'fugu__text__number_of_sentences_retelling':sum(np.array(max_similarities_hallucinations) < .5)/len(max_similarities_hallucinations)})

    if distances.empty:
        distances = pd.DataFrame(max_similarities_dict,index=[0])
    else:
        distances = pd.concat((distances,pd.DataFrame(max_similarities_dict,index=[0])),axis=0)
        
distances.to_csv(Path(data_dir,f'distances_{model_name.split("/")[1]}.csv'),encoding='utf-16',index=False,sep=',')