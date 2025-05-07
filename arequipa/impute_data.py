from sklearn.impute import KNNImputer
import pandas as pd

imputer = KNNImputer(n_neighbors=5)

data = pd.read_excel(r"/Users/gp/Downloads/Features_Marcio_imagenes_ordenado.xlsx")

id = data.pop('id')

imputed_data = pd.DataFrame(data=imputer.fit_transform(data),columns=data.columns)
imputed_data['id'] = id

imputed_data.to_csv(r"/Users/gp/Downloads/data_marcio_imputada.csv")