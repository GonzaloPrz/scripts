import pandas as pd
from pathlib import Path
import re

df = pd.read_csv(Path(Path.home(),'data','sj','transcripts.csv'))
features = pd.read_csv(Path(Path.home(),'data','sj','all_features.csv'))

# Function to count words excluding evaluator interventions
def count_participant_words(transcript):
    # Remove evaluator interventions
    participant_text = re.sub(r'\[.*?\]', '', transcript)
    return len(participant_text.split())

# Function to count words in evaluator interventions
def count_evaluator_words(transcript):
    # Extract evaluator interventions
    evaluator_texts = re.findall(r'\[.*?\]', transcript)
    evaluator_text = ' '.join(evaluator_texts)
    return len(evaluator_text.split())

# Calculate the number of words for each participant and evaluator
df['participant_words'] = df['revised_text'].apply(count_participant_words)
df['evaluator_words'] = df['revised_text'].apply(count_evaluator_words)

# Group by task
grouped = df.groupby('task')

filtered_participants_list = []

for task, group in grouped:

    outliers = group[group['participant_words'] < 5]
    # Filter out participants who produce less than 3 times the amount of words as the evaluator
    less_than_evaluator = group[group['participant_words'] < (9*group['evaluator_words'])]

    # Combine the two conditions
    filtered_participants = pd.concat([outliers, less_than_evaluator]).drop_duplicates()
    filtered_participants_list.append(filtered_participants)

# Combine all filtered participants from each task
final_filtered_participants = pd.concat(filtered_participants_list).drop_duplicates()

# Save the filtered participants to a new CSV file
final_filtered_participants.to_csv(Path(Path.home(),'data','sj','filtered_transcripts.csv'), index=False)
kept_participants = df[~df.index.isin(final_filtered_participants.index)]
kept_participants.to_csv(Path(Path.home(),'data','sj','kept_transcripts.csv'), index=False)
kept_features = features[features['id'].isin(kept_participants['id'])]
kept_features.to_csv(Path(Path.home(),'data','sj','kept_features.csv'), index=False)
