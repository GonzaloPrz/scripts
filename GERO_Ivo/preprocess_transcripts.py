import pandas as pd
from pathlib import Path

df_transcripts = pd.read_excel(Path(Path(__file__).parent,'data','valid_responses.xlsx'))

df_modified = pd.DataFrame()

tasks = ['letra_f','letra_a','letra_s','animales']

for task in tasks:
    df_task = df_transcripts[df_transcripts['Task'] == task]
    df_task[f'{task}_valid_responses'] = df_task['Valid_Responses_count']
    df_task.drop(columns=['Task','Valid_Responses_count'], inplace=True)

    if df_modified.empty:
        df_modified = df_task.reset_index(drop=True)
    else:
        df_modified = pd.merge(df_modified, df_task, on='Codigo')

df_modified.to_excel(Path(Path(__file__).parent,'data','valid_responses_mod.xlsx'), index=False)