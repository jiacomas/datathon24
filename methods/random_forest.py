import pandas as pd
from sklearn.preprocessing import LabelEncoder

bodegon = pd.read_csv('/Users/jia/Documents/datathon24/bodegon.csv') 

numeric_bodegon = bodegon.apply(pd.to_numeric)
encoder = LabelEncoder()
numeric_bodegon["des_sex"] = encoder.fit_transform(bodegon["des_sex"])


