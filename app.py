import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd

class FixedDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=None):
        if not training:
            return inputs
        noise_shape = self.noise_shape if self.noise_shape is not None else tf.shape(inputs)
        return tf.nn.dropout(inputs, rate=self.rate, noise_shape=noise_shape, seed=self.seed)

# Registrar capas personalizadas al cargar el modelo
with tf.keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
    model = tf.keras.models.load_model('my_model.h5')  # Asegúrate de que la ruta sea correcta

# Streamlit Interface
st.title("TheInterns")

st.markdown("""
### Mango challenge
This is our streamlit interface for the challenge. The rules are simple:
1. Drag and drop an image.
2. Push the button "Find the attributes"
3. Here you have your attributes for your selected clothe!
""")

def prepare_image(image):
    # Convertir a modo RGB si no lo está
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Redimensionar la imagen al tamaño adecuado para el modelo (ajusta según lo que necesite tu modelo)
    image = image.resize((224, 224))  # Ajusta 224x224 si tu modelo usa este tamaño
    image = np.array(image)  # Convertir la imagen a un array numpy
    image = np.expand_dims(image, axis=0)  # Añadir una dimensión para el batch
    image = image / 255.0  # Normalizar la imagen
    return image

attribute_columns = ['silhouette_type', 'neck_lapel_type', 'woven_structure', 'knit_structure',
                     'heel_shape_type', 'length_type', 'sleeve_length_type', 'toecap_type',
                     'waist_type', 'closure_placement', 'cane_height_type']

uploaded_image = st.file_uploader("Select an image", type=["jpg", "png", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Image uploaded", use_column_width=True)

    if st.button("Find the attributes"):
        prepared_image = prepare_image(image)
        prediction = model.predict(prepared_image)
        predictions_mapped = []
        for prediction_ in prediction:
            labels_mapped = []
            for label in prediction_:  # Use enumerate to track index in prediction
                max_idx = np.argmax(label)  # Find the index of the max value in the label
                if label[max_idx] > 0.85:
                    labels_mapped.append(int(max_idx))  # Set the value at the current position to max_idx
                else:
                    labels_mapped.append(int(0))
            predictions_mapped.append(labels_mapped)
        predictions_df = pd.DataFrame(predictions_mapped, columns=attribute_columns)
        attributes_df = pd.read_csv("./archive/attribute_data.csv")
        attributes_dict = {}
        # Iterate over each row and create the key-value pairs
        for _, row in attributes_df.iterrows():
            key = f"{row['attribute_name']}_{row['cod_value']}"
            value = row['des_value']
            attributes_dict[key] = value
        
        for column in predictions_df.columns:
            predictions_df[column] = predictions_df[column].apply(lambda x: attributes_dict.get(f"{column}_{x}", 'INVALID'))

        st.dataframe(predictions_df)
else:
    st.write("Please, upload an image")
