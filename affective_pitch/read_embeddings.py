import pickle
from pathlib import Path
import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd

data_dir = Path(Path.home(), 'data', 'affective_pitch') if '/Users/gp' in str(Path.home()) else Path('D:', 'CNC_Audio', 'gonza', 'data', 'affective_pitch')

segmentations = ['windows','phrases']

for segmentation in segmentations:
    embeddings_df = pd.DataFrame()
    embeddings_df_pos = pd.DataFrame()
    embeddings_df_neu = pd.DataFrame()
    embeddings_df_neg = pd.DataFrame()

    all_embeddings = pickle.load(open(data_dir / f'embeddings_{segmentation}.pkl','rb'))
    mean_embeddings = pd.DataFrame(columns=['audio_path','mean_embedding'])
    mean_embeddings_pos = pd.DataFrame(columns=['audio_path','mean_embedding'])
    mean_embeddings_neu = pd.DataFrame(columns=['audio_path','mean_embedding'])
    mean_embeddings_neg = pd.DataFrame(columns=['audio_path','mean_embedding'])

    if segmentation == 'windows':
        all_embeddings['audio_path'] = all_embeddings['audio_path'].apply(lambda x: 'REDLAT_' + x.replace('__','_'))
    audio_paths = np.unique([audio_path for audio_path in all_embeddings['audio_path']])

    ids = np.unique([audio_path.split('_')[1] for audio_path in audio_paths])
    
    for id in ids:
        paths_for_current_id = [audio_path for audio_path in audio_paths if id in audio_path]

        row = {'id':id}
        row_pos = {'id':id}
        row_neu = {'id':id}
        row_neg = {'id':id}

        for audio_path in paths_for_current_id:
            
            try:
                embedding = np.nanmean([emb for audio_path_, emb in zip(all_embeddings['audio_path'], all_embeddings['z']) if audio_path_ == audio_path],axis=0)
            except:
                continue

            mean_embeddings.loc[len(mean_embeddings),:] = [audio_path,embedding.squeeze()]

            if embedding.size == 0:
                continue
            
            segment = '_'.join(audio_path.split('_')[-2:]).replace('.wav','')

            row.update({
                    f'Fugu__{segment}__embedding_mean': np.nanmean(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_std': np.nanstd(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_min': np.nanmin(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_max': np.nanmax(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_kurtosis': kurtosis(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_skew': skew(embedding.squeeze())
                })

            if Path(data_dir,'divided_audios',segmentation,'POS',f'REDLAT_{id}_Fugu_{segment}.wav').exists() or Path(data_dir,'divided_audios',segmentation,'POS',f'{id}_Fugu__{segment}.wav').exists():
                row_pos.update({
                    f'Fugu__{segment}__embedding_mean': np.nanmean(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_std': np.nanstd(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_min': np.nanmin(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_max': np.nanmax(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_kurtosis': kurtosis(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_skew': skew(embedding.squeeze())
                })
                mean_embeddings_pos.loc[len(mean_embeddings_pos),:] = [audio_path,embedding.squeeze()]

            elif Path(data_dir,'divided_audios',segmentation,'NEU',f'REDLAT_{id}_Fugu_{segment}.wav').exists() or Path(data_dir,'divided_audios',segmentation,'NEU',f'{id}_Fugu__{segment}.wav').exists():
                row_neu.update({
                    f'Fugu__{segment}__embedding_mean': np.nanmean(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_std': np.nanstd(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_min': np.nanmin(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_max': np.nanmax(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_kurtosis': kurtosis(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_skew': skew(embedding.squeeze())
                })
                mean_embeddings_neu.loc[len(mean_embeddings_neu),:] = [audio_path,embedding.squeeze()]
            elif Path(data_dir,'divided_audios',segmentation,'NEG',f'REDLAT_{id}_Fugu_{segment}.wav').exists() or Path(data_dir,'divided_audios',segmentation,'NEG',f'{id}_Fugu__{segment}.wav').exists():
                row_neg.update({
                    f'Fugu__{segment}__embedding_mean': np.nanmean(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_std': np.nanstd(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_min': np.nanmin(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_max': np.nanmax(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_kurtosis': kurtosis(embedding.squeeze()),
                    f'Fugu__{segment}__embedding_skew': skew(embedding.squeeze())
                })
                mean_embeddings_neg.loc[len(mean_embeddings_neg),:] = [audio_path,embedding.squeeze()]
                
        if embeddings_df.empty:
            embeddings_df = pd.DataFrame(columns=row.keys())
        
        if embeddings_df_pos.empty:
            embeddings_df_pos = pd.DataFrame(columns=row_pos.keys())
        if embeddings_df_neu.empty:
            embeddings_df_neu = pd.DataFrame(columns=row_neu.keys())
        if embeddings_df_neg.empty:
            embeddings_df_neg = pd.DataFrame(columns=row_neg.keys())
        
        if len(row_pos) > 1:
            embeddings_df_pos = pd.concat([embeddings_df_pos, pd.DataFrame([row_pos])],axis=0, ignore_index=True)
        if len(row_neu) > 1:
            embeddings_df_neu = pd.concat([embeddings_df_neu, pd.DataFrame([row_neu])],axis=0, ignore_index=True)
        if len(row_neg) > 1:
            embeddings_df_neg = pd.concat([embeddings_df_neg, pd.DataFrame([row_neg])],axis=0, ignore_index=True)

        embeddings_df = pd.concat([embeddings_df, pd.DataFrame([row])], axis=0, ignore_index=True)

    embeddings_df_pos.to_csv(Path(data_dir / 'divided_audios' / segmentation / 'POS', f'POS_embeddings_features_{segmentation}.csv'), index=False)
    embeddings_df_neu.to_csv(Path(data_dir / 'divided_audios' / segmentation / 'NEU', f'NEU_embeddings_features_{segmentation}.csv'), index=False)
    embeddings_df_neg.to_csv(Path(data_dir / 'divided_audios' / segmentation / 'NEG', f'NEG_embeddings_features_{segmentation}.csv'), index=False)
    embeddings_df.to_csv(Path(data_dir / 'divided_audios' / segmentation / 'ALL', f'ALL_embeddings_features_{segmentation}.csv'), index=False)
    pickle.dump(embeddings_df,open(Path(data_dir / 'divided_audios' / segmentation / 'ALL', f'ALL_mean_embeddings_{segmentation}.pkl'),'wb'))
    pickle.dump(embeddings_df_pos,open(Path(data_dir / 'divided_audios' / segmentation / 'POS', f'POS_mean_embeddings_{segmentation}.pkl'),'wb'))
    pickle.dump(embeddings_df_neu,open(Path(data_dir / 'divided_audios' / segmentation / 'NEU', f'NEU_mean_embeddings_{segmentation}.pkl'),'wb'))
    pickle.dump(embeddings_df_neg,open(Path(data_dir / 'divided_audios' / segmentation / 'NEG', f'NEG_mean_embeddings_{segmentation}.pkl'),'wb'))
    