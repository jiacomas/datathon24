import pandas as pd

# Carregar el CSV amb les dades
file = '/Users/jia/Documents/datathon24/archive/test_data.csv'  # Substitueix amb el camí del teu arxiu
data = pd.read_csv(file)

# Comprovar les primeres files del dataset per verificar
print(data.head())

# Agrupar les dades per la columna 'attribute_name'
grouped_data = data.groupby('attribute_name')

# Iterar per cada grup i escriure un arxiu CSV per a cada valor de 'attribute_name'
for label, group in grouped_data:
    # Crear un nom de fitxer basat en el valor del label
    output_file = f'{label}_data.csv'  # El nom del fitxer serà el valor de 'attribute_name'
    
    # Escriure el grup en un nou fitxer CSV
    group.to_csv(output_file, index=False)  # index=False per no guardar l'índex com a columna
    print(f"Fitxer '{output_file}' creat amb {len(group)} files.")
