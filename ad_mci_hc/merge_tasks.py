from pathlib import Path
import pandas as pd

data_dir = Path("/Users/gp/data/ad_mci_hc_ct")

data_pleasant_memory = pd.read_csv(data_dir / "data_pleasant-memory.csv")
data_routine = pd.read_csv(data_dir / "data_routine.csv")
data_routine.drop(columns=['sex','age','education','group'], inplace=True)

data_image = pd.read_csv(data_dir / "data_image.csv")
data_image.drop(columns=['sex','age','education','group'], inplace=True)

data_pleasant_memory_routine = pd.merge(data_pleasant_memory, data_routine, how='inner', on='id')
data_image_pleasant_memory_routine = pd.merge(data_image,data_pleasant_memory_routine, how='inner', on='id')

data_image_pleasant_memory = pd.merge(data_image, data_pleasant_memory, how='inner', on='id')
data_image_routine = pd.merge(data_image, data_routine, how='inner', on='id')

data_pleasant_memory_routine = pd.merge(data_pleasant_memory,data_routine, how='inner', on='id')

data_image_pleasant_memory_routine.to_csv(data_dir / "data_image_pleasant_memory_routine.csv", index=False)
data_image_pleasant_memory.to_csv(data_dir / "data_image_pleasant_memory.csv", index=False)
data_image_routine.to_csv(data_dir / "data_image_routine.csv", index=False)
data_pleasant_memory_routine.to_csv(data_dir / "data_pleasant_memory_routine.csv", index=False)

