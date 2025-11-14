import pandas as pd
from pathlib import Path

data_dir = Path("/Users/gp/data/arequipa_reg")

#composite_matched = pd.read_csv(data_dir / "composite_matched.csv")
#all_data_matched = pd.read_csv(data_dir / "data_matched_group.csv")
#composite_matched = pd.merge(composite_matched, all_data_matched, on='id', how='inner')

#composite_matched.to_csv(data_dir / "data_matched_group.csv", index=False)
composite_unmatched = pd.read_csv(data_dir / "composite_unmatched.csv")
all_data_unmatched = pd.read_csv(data_dir / "data_unmatched_group.csv")
composite_unmatched = pd.merge(composite_unmatched, all_data_unmatched, on='id', how='inner')

composite_unmatched.to_csv(data_dir / "data_unmatched_group.csv", index=False)
#all_data = pd.concat((composite_matched, composite_unmatched), ignore_index=True)

#all_data.to_csv(Path(data_dir,'composite_scores.csv'),index=False)
#data_imagenes = pd.read_csv(data_dir / "data_imagenes_matched_group.csv")['id']

#composite_scores_imagenes = pd.merge(all_data, data_imagenes, on='id', how='inner')

#composite_scores_imagenes.to_csv(Path(data_dir,'composite_scores_imagenes.csv'),index=False)
