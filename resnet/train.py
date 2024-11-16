import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import numpy as np

# Cargar el archivo CSV con las características
df = pd.read_csv('wide_data.csv')

# Definir la carpeta donde se encuentran las imágenes
image_folder = 'images/images'

# Seleccionar un subconjunto de imágenes (por ejemplo, las primeras 100 filas del DataFrame)
df_subset = df.head(999)  # Cambia el número de filas según lo que necesites

# Función para cargar las imágenes y las etiquetas correspondientes
def load_images_and_labels(df, image_folder):
    images = []
    labels = []

    for index, row in df.iterrows():
        img_id = row['des_filename']
        img_path = os.path.join(image_folder, f"{img_id}")

        # Cargar y preprocesar la imagen
        try:
            img = load_img(img_path, target_size=(224, 224))  # Redimensionar la imagen
            img_array = img_to_array(img)  # Convertir la imagen a array
            img_array = preprocess_input(img_array)  # Preprocesar la imagen para ResNet50
            images.append(img_array)
        except Exception as e:
            print(f"Error al cargar la imagen {img_path}: {e}")
            continue

        # Extraer las etiquetas de las características
        label = [
            row['silhouette_type'],
            row['waist_type'],
            row['neck_lapel_type'],
            row['sleeve_length_type'],
            row['toecap_type'],
            row['closure_placement'],
            row['cane_height_type'],
            row['heel_shape_type'],
            row['knit_structure'],
            row['length_type'],
            row['woven_structure']
        ]

        # Rellenar valores nulos con una categoría predeterminada, como 0
        label = [0 if pd.isna(x) else int(x) for x in label]
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Convertir las etiquetas en formato one-hot (por cada columna de etiquetas)
    labels_one_hot = [to_categorical(labels[:, i], num_classes=10000) for i in range(labels.shape[1])]

    return images, labels_one_hot

# Cargar solo las imágenes y etiquetas del subconjunto
images, labels_one_hot = load_images_and_labels(df_subset, image_folder)

# Crear el modelo base de ResNet50 sin la capa superior
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Agregar las capas para la predicción de las características
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Añadir una capa de salida para cada característica
output_silhouette = Dense(10000, activation='softmax', name='silhouette_type')(x)
output_waist = Dense(10000, activation='softmax', name='waist_type')(x)
output_neck_lapel = Dense(10000, activation='softmax', name='neck_lapel_type')(x)
output_sleeve_length = Dense(10000, activation='softmax', name='sleeve_length_type')(x)
output_toecap = Dense(10000, activation='softmax', name='toecap_type')(x)
output_closure = Dense(10000, activation='softmax', name='closure_placement')(x)
output_cane_height = Dense(10000, activation='softmax', name='cane_height_type')(x)
output_heel_shape = Dense(10000, activation='softmax', name='heel_shape_type')(x)
output_knit = Dense(10000, activation='softmax', name='knit_structure')(x)
output_length = Dense(10000, activation='softmax', name='length_type')(x)
output_woven = Dense(10000, activation='softmax', name='woven_structure')(x)

# Crear el modelo con múltiples salidas
model = Model(inputs=base_model.input, outputs=[output_silhouette, output_waist, output_neck_lapel,
                                                output_sleeve_length, output_toecap, output_closure,
                                                output_cane_height, output_heel_shape, output_knit,
                                                output_length, output_woven])

# Congelar las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] * 11)

# Entrenar el modelo con las imágenes y etiquetas seleccionadas
model_saved = model.fit(
    images,
    labels_one_hot,
    epochs=10,
    batch_size=16  # Ajusta el tamaño del batch según la memoria disponible
)

model.save('resnet.h5')
