from datetime import datetime
import pandas as pd

atribute_data = pd.read_csv('archive/attribute_data.csv')
product_data = pd.read_csv('archive/product_data.csv')

pivoted_attributes = atribute_data.pivot_table(index='cod_modelo_color', columns='attribute_name', values='des_value', aggfunc='first').reset_index()
product_data2 = pd.merge(product_data, pivoted_attributes, on='cod_modelo_color', how='left')

#TODO consider the cod_value

for i in range(len(product_data2)):
    if "_B" in product_data.iloc[i]["des_filename"]:
        product_data2.iloc[i].to_frame().T.to_csv('bodegon.csv', mode='a', header=False, index=False)
    else:
        product_data2.iloc[i].to_frame().T.to_csv('no_bodegon.csv', mode='a', header=False, index=False)
