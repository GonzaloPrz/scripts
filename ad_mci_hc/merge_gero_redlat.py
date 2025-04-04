import pandas as pd
from pathlib import Path

redlat_data = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','data_matched_group_AD_HC.csv'))
gero_data = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','all_data_gero.csv'))

features = list(set(redlat_data.columns) & set(gero_data.columns))

dem_vars = [ft for ft in features if '__' not in ft]
feature_vars = [ft for ft in features if '__' in ft]

redlat_data = redlat_data[dem_vars + feature_vars]
gero_data = gero_data[dem_vars + feature_vars]

#Keep only the subjects with id containing COH or SL

redlat_data.group = redlat_data.group.replace({'CN':'HC','AD':'AD'})
#redlat_data = redlat_data.dropna(subset=[col for col in redlat_data.columns if 'fugu__' in col],how='all')

all_data = pd.concat([redlat_data,gero_data],ignore_index=True)
all_data = all_data.dropna(subset=[col for col in all_data.columns if 'fugu__' in col],how='all')

all_data.to_csv(Path(Path.home(),'data','ad_mci_hc','all_data_2.csv'),index=False)