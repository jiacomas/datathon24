import pandas as pd

# Leer el archivo con predicciones
predictions_file = 'predictions.csv'
predictions_df = pd.read_csv(predictions_file)

# Limpiar el ID principal y atributo
predictions_df['test_id'] = predictions_df['test_id'].str.extract(r'(.*?_[0-9]+)__(.*)')[0] + "_" + \
                            predictions_df['test_id'].str.extract(r'(.*?_[0-9]+)__(.*)')[1]

# Leer el archivo con las codificaciones
encoding_file = '../archive/attribute_data.csv'
encoding_df = pd.read_csv(encoding_file)

# Crear el diccionario de codificaciones
encoding_dict = {}
for _, row in encoding_df.iterrows():
    attribute_name = row['attribute_name']
    cod_value = f"{int(row['cod_value']):04d}"  # Asegurarse de que el valor tenga formato 0000
    des_value = row['des_value']

    if attribute_name not in encoding_dict:
        encoding_dict[attribute_name] = {}
    
    encoding_dict[attribute_name][cod_value] = des_value

# Función para convertir valores numéricos en descripciones
def convert_to_description(row):
    attribute = row['test_id'].split('_')[-1]  # Extraer el atributo
    cod_value = row['des_value']

    if cod_value == "0000":
        return "INVALID"  # Reemplazar con INVALID si el código es 0000
    return encoding_dict.get(attribute, {}).get(cod_value, "UNKNOWN")  # Devolver descripción o UNKNOWN

# Aplicar la conversión al dataframe
predictions_df['des_value'] = predictions_df.apply(convert_to_description, axis=1)

# Guardar el archivo de salida
output_file = 'submission.csv'
predictions_df.to_csv(output_file, index=False)

print(f"Archivo procesado y guardado en {output_file}.")
