import os
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import UnidentifiedImageError

# Ruta de la carpeta de imágenes
test_image_folder = 'images/images'

# Obtener la lista de archivos en la carpeta
all_images = sorted(
    [f for f in os.listdir(test_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
)[-4000:]  # Seleccionar las últimas 4000 imágenes

# Función para cargar y preprocesar una imagen
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Redimensionar
        img_array = img_to_array(img)  # Convertir a array
        img_array = preprocess_input(img_array)  # Preprocesar
        return img_array
    except UnidentifiedImageError:
        print(f"Advertencia: No se pudo cargar la imagen {image_path}")
        return None

# Filtrar imágenes válidas
test_images_arrays = []
valid_image_paths = []

for img in all_images:
    img_path = os.path.join(test_image_folder, img)
    processed_img = preprocess_image(img_path)
    if processed_img is not None:
        test_images_arrays.append(processed_img)
        valid_image_paths.append(img)  # Solo guardar rutas de imágenes válidas

test_images_arrays = np.array(test_images_arrays)

# Cargar el modelo previamente entrenado
model = load_model('resnet.h5')

# Realizar predicciones
predictions = model.predict(test_images_arrays)

# Crear el archivo CSV para guardar los resultados
output_csv = 'predictions.csv'

# Escribir los resultados en el archivo CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['test_id', 'des_value'])  # Escribir cabecera del archivo CSV

    for img_idx, img_id in enumerate(valid_image_paths):
        img_name = os.path.splitext(img_id)[0]  # Obtener el ID de la imagen sin extensión
        for output_idx, output_name in enumerate(model.output_names):
            class_id = np.argmax(predictions[output_idx][img_idx])  # Clase más probable
            writer.writerow([f"{img_name}_{output_name}", f"{class_id:04d}"])  # Escribir la fila con formato numérico
