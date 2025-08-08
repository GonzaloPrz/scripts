from pathlib import Path
import sys,tqdm,itertools,json,importlib
import pandas as pd
from warnings import filterwarnings

filterwarnings('ignore')

subsets = ['audio']

dimensions = {'audio':['talking_intervals','pitch_analysis']
        }

file_extentions = {'audio':'.wav'}

project_name = 'sj'

for subset in subsets:
    for dimension in dimensions[subset]:
        
        module_path = Path(Path.home(), 'tell', 'local_feature_extraction', f'{subset}_features', dimension)
        sys.path.append(str(module_path))
        
        # Dynamically import the app module
        try:
            app = importlib.import_module('app')
            importlib.reload(app)
        except:
            sys.path.append(str(module_path))

            app = importlib.import_module('app')
            importlib.reload(app)

        data_dir = Path(Path.home(),'data',project_name,'audios_prepro')

        extracted_features = pd.DataFrame()
        files = [file for file in data_dir.iterdir() if file.suffix == file_extentions[subset]]

        for file in tqdm.tqdm(files):
            id = '_'.join(file.stem.split('_')[:2])
            task = file.stem.split('_')[2]

            if not Path(data_dir.parent,'transcripciones',f'{id}_{task}_mono_16khz_diarize_loudnorm_denoised.txt').exists():
                continue
            
            try:
                features_dict = app.main(str(file))
                if 'data' in features_dict.keys():
                    features = features_dict['data'].copy()
                else:
                    features_dict = json.loads(features_dict['body']).copy()
                    features = features_dict['data'].copy()

            except:
                print(f'Error extracting features from {file.stem}')
                continue
            features.update(features_dict['scores'])
            features = {f'{dimension}__{k}': v for k, v in features.items() if not isinstance(v, list)}

            features['id'] = id
            features['task'] = task 

            if extracted_features.empty:
                extracted_features = pd.DataFrame(features, index=[0])
            else:
                extracted_features.loc[len(extracted_features)] = features.copy()
            
        #Remove app from sys.path
        sys.path.remove(str(module_path))

        extracted_features['task'] = extracted_features['task'].apply(lambda x: x.replace('Preg','Pre').replace('pre','Pre').replace('post','Post').replace('2Pos1','2Post1').replace('2Pos2','2Post2'))

        extracted_features.to_csv(Path(Path.home(),'data',project_name,f'{dimension}_features.csv'),index=False)
        del extracted_features, features, features_dict