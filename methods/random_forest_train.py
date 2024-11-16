import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

# Función para preprocesar datos y convertir categorías a valores numéricos
def get_numeric_data(file, categorical_columns, encoder=LabelEncoder()):
    bodegon = pd.read_csv(file)
    
    # Rellenar valores faltantes con "INVALID"
    bodegon[categorical_columns] = bodegon[categorical_columns].fillna("INVALID")
    
    # Convertir columnas categóricas a cadenas
    for column in categorical_columns:
        bodegon[column] = bodegon[column].astype(str)
    
    # Codificar todas las columnas categóricas
    for column in categorical_columns:
        bodegon[column] = encoder.fit_transform(bodegon[column])
    
    return bodegon

if __name__ == "__main__":
    # Archivo de entrada
    file = '/Users/luke/datathon24/bodegon.csv'
    labels = ['silhouette_type', 'neck_lapel_type', 'woven_structure', 'knit_structure', 'heel_shape_type', 'length_type', 
              'sleeve_length_type', 'toecap_type', 'waist_type', 'closure_placement', 'cane_height_type']
    
    # Preprocesar los datos
    data_clean = get_numeric_data(file, ['des_color', 'des_sex', 'des_age', 'des_line', 'des_fabric', 
        'des_product_category', 'des_product_aggregated_family', 'des_product_family', 
        'des_product_type', 'silhouette_type', 'neck_lapel_type', 'woven_structure',
        'knit_structure', 'heel_shape_type', 'length_type', 'sleeve_length_type',
        'toecap_type', 'waist_type', 'closure_placement', 'cane_height_type'])
    
    # Dividir datos en características (X) y múltiples objetivos (Y)
    X = data_clean.drop(labels + ["cod_modelo_color", "des_filename", "cod_color"], axis=1)
    Y = data_clean[labels]  # Mantener las columnas de etiquetas como un DataFrame
    print(X.head())
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=29)
    
    # Crear el modelo MultiOutput con RandomForestClassifier
    base_model = RandomForestClassifier(n_estimators=100, random_state=29, class_weight='balanced')
    multioutput_model = MultiOutputClassifier(base_model)
    
    # Entrenar el modelo
    multioutput_model.fit(X_train, Y_train)
    
    # Hacer predicciones
    Y_pred = multioutput_model.predict(X_test)
    
    # Calcular precisión promedio para todas las etiquetas
    accuracies = [accuracy_score(Y_test.iloc[:, i], Y_pred[:, i]) for i in range(len(labels))]
    print("Precisión global promedio:", sum(accuracies) / len(accuracies))
    
    # Leer el archivo de codificación para mapear predicciones
    encoding = pd.read_csv("/Users/luke/datathon24/archive/attribute_data.csv")
    encoding_dict = {}

    for _, row in encoding.iterrows():
        attribute_name = row['attribute_name']
        cod_value = row['cod_value']
        des_value = row['des_value']
        
        if attribute_name not in encoding_dict:
            encoding_dict[attribute_name] = {}
        
        encoding_dict[attribute_name][cod_value] = des_value

    # Preparar DataFrame para resultados
    resultados = pd.DataFrame(columns=["test_id", "des_value"])
    for i, label in enumerate(labels):
        # Crear DataFrame temporal con predicciones decodificadas
        temp_results = pd.DataFrame({
            "test_id": data_clean.loc[X_test.index, "cod_modelo_color"].astype(str) + "_" + label,
            "des_value": Y_pred[:, i]  # Obtener las predicciones para esta etiqueta
        })
        # Mapear valores codificados a sus descripciones
        temp_results['des_value'] = temp_results['des_value'].map(encoding_dict[label])
        temp_results['des_value'].fillna('INVALID', inplace=True)
        
        # Concatenar al DataFrame final
        resultados = pd.concat([resultados, temp_results], ignore_index=True)
    
    # Guardar los resultados en un archivo CSV
    resultados.to_csv("predicciones_multioutput.csv", index=False)
    print("Predicciones guardadas en 'predicciones_multioutput.csv'")
    # Usar el modelo entrenado para predecir sobre un nuevo dataset sin etiquetas esperadas


#-------------------------------------------------------------

# Predecir sobre el dataset real
file_real = '/Users/luke/datathon24/archive/test_data.csv'  # Reemplaza con la ruta de tu dataset real

# Columnas categóricas esperadas en test_data
categorical_columns_real = ["cod_color", "des_sex", "des_age", "des_line", "des_fabric",
                            "des_product_category", "des_product_aggregated_family", "des_product_family",
                            "des_product_type", "des_filename", "des_color"]

# Preprocesar datos reales
data_real = get_numeric_data(file_real, categorical_columns_real)

# Inicializar DataFrame para resultados
resultados_real = pd.DataFrame(columns=["test_id", "predicted_value"])

# Iterar sobre cada fila y predecir dinámicamente
for index, row in data_real.iterrows():
    attribute_name = row["attribute_name"]  # Etiqueta que debe predecirse
    if attribute_name not in labels:
        print(f"Advertencia: '{attribute_name}' no es una etiqueta válida. Ignorando fila {index}.")
        continue

    # Preparar los datos para esta fila
    X_row = data_real.loc[[index]].drop(["cod_modelo_color", "des_filename", "cod_color", "attribute_name", "test_id"], axis=1)
    # Mover la columna 1 a la última posición
    cols = X_row.columns.tolist()
    cols.append(cols.pop(0))
    X_row = X_row[cols]
    print(index)
    # Predecir para la etiqueta específica
    label_index = labels.index(attribute_name)
    predicted_value_code = multioutput_model.estimators_[label_index].predict(X_row)[0]

    # Decodificar el valor predicho
    predicted_value = encoding_dict[attribute_name].get(predicted_value_code, "INVALID")
    
    # Agregar los resultados al DataFrame
    resultados_real = pd.concat([resultados_real, pd.DataFrame({
        "test_id": [row["test_id"]],
        "des_value": [predicted_value]
    })], ignore_index=True)

# Guardar las predicciones en un archivo CSV
resultados_real.to_csv("entrega.csv", index=False)
print("Predicciones guardadas en 'entrega.csv'")

