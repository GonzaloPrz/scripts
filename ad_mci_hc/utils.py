from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import pandas as pd
import ast
import nltk

from nltk.tokenize import sent_tokenize


@dataclass
class SentenceEmbedding:
    sentence: str
    embedding: List[float]

@dataclass
class VideoSegment:
    description_text: str
    sentence_embeddings: List[SentenceEmbedding] = field(default_factory=list)

    def compute_sentence_embeddings(self, model: SentenceTransformer):
        """Tokenize description text into sentences and embed each."""
        sentences = sent_tokenize(self.description_text, language='spanish')
        embeddings = model.encode(sentences)
        self.sentence_embeddings = [
            SentenceEmbedding(sentence=s, embedding=e.tolist()) for s, e in zip(sentences, embeddings)
        ]

    @classmethod
    def from_row(cls, row: pd.Series,text_column: str) -> 'VideoSegment':
        """Factory method to create VideoSegment from a pandas row."""
        description = row[text_column]
        return cls(
            description_text=description.get('description', '') if isinstance(description, dict) else description
        )
    
def eval_or_return(x):
    if pd.notnull(x) and x.strip().startswith("{") and x.strip().endswith("}"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    else:
        return x
    
def load_video_information(df: pd.DataFrame,text_column:str="Description") -> List[VideoSegment]:    
    
    # Parse the Description field to dict
    if isinstance(df,pd.DataFrame):
        df[text_column] = df[text_column].apply(eval_or_return)
    
        segments = [VideoSegment.from_row(row,text_column) for _, row in df.iterrows()]
    else:
        segments = [VideoSegment.from_row(df,text_column)]
    return segments
