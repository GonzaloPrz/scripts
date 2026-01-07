import pandas as pd
from pathlib import Path

def filter_data(data,criteria):
    for criterion,value in criteria.items():
        data = data.loc[data[criterion].isin(value)]
        
        data = data.drop(criterion,axis=1)
    
    data = data.drop([col for col in data.columns if any(x in col for x in ['has_','Unnamed'])],axis=1)
    return data

data_dir = Path(Path.home(),'data','crossling_mci') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','crossling_mci')

all_data = pd.read_csv(Path(data_dir,'all_data.csv'))

data_french_pos = filter_data(all_data,{'language':['fr-FR'],'story.type':['positive'],'group':[1]})

data_french_pos.to_csv(Path(data_dir,'data_french_positive_mci.csv'),index=None)

data_french_neg = filter_data(all_data,{'language':['fr-FR'],'story.type':['negative'],'group':[1]})
data_french_neg.to_csv(Path(data_dir,'data_french_negative_mci.csv'),index=None)
               
data_spanish_pos = filter_data(all_data,{'language':['es-ES'],'story.type':['positive'],'group':[1]})
data_spanish_pos.to_csv(Path(data_dir,'data_spanish_positive_mci.csv'),index=None)

data_spanish_neg = filter_data(all_data,{'language':['es-ES'],'story.type':['negative'],'group':[1]})
data_spanish_neg.to_csv(Path(data_dir,'data_spanish_negative_mci.csv'),index=None)

