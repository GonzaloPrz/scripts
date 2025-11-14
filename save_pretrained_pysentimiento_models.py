import os
from pysentimiento import create_analyzer
import pickle
repos = {
        "es": "robertuito-sentiment-analysis",
        "pt": "bertweet-pt-sentiment",
        "en": "bertweet-base-sentiment-analysis",
        "it": "bert-it-sentiment"
}

os.makedirs("models", exist_ok=True)

for lang in repos.keys():
    os.makedirs(f"models/sentiment-{lang}", exist_ok=True)
    sent_analyzer = create_analyzer(task="sentiment", lang=lang)
    if hasattr(sent_analyzer, "model"):
        sent_analyzer.model.to("cpu")

    with open(f"models/sentiment-{lang}/sent_analyzer.pkl", "wb") as f:
        pickle.dump(sent_analyzer, f)

    print(f"✅ saved models/sentiment-{lang}")
