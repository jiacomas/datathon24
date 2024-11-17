import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# train_path = os.path.join(parent_dir, '/clean_data/bodegon.csv')
# test_path = os.path.join(parent_dir, '/archive/test_data.csv')
train_path = '/Users/jia/Documents/datathon24/clean_data/bodegon.csv'
test_path = '/Users/jia/Documents/datathon24/archive/test_data.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

image_size = (128, 128)  
train_images = []
train_labels = []

for _, row in df_train.iterrows():
    image_path = row['des_filename']
    path = f'/Users/jia/Documents/datathon24/archive/images/images/{image_path}'
    if os.path.exists(path):
        img = load_img(path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  
        train_images.append(img_array)
        train_labels.append(row['cod_modelo_color'])

train_images = np.array(train_images)
train_labels = np.array(train_labels)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(np.unique(train_labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

model.fit(train_images, train_labels_encoded, epochs=10, batch_size=32)
loss, accuracy = model.evaluate(train_images, train_labels_encoded)
print(f'Accuracy obtingut: {accuracy * 100:.2f}%')


type_path = 'knit_structure'
test_path = f'/Users/jia/Documents/datathon24/divide/{type_path}_data.csv'
df_test = pd.read_csv(test_path)

test_results = []

for _, row in df_test.iterrows():
    image_path = row['des_filename']
    path = f'/Users/jia/Documents/datathon24/archive/images/images/{image_path}'
    try:
        if os.path.exists(path):
            img = load_img(path, target_size=image_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  
            prediction = model.predict(img_array)
            predicted_label = np.argmax(prediction)
            
            des_value = row[row['attribute_name']]
        else:
            des_value = 'INVALID'
    except Exception:
        des_value = 'EEEEE'
    
    test_results.append({
        'test_id': row['test_id'],
        'des_value': des_value
    })

df_results = pd.DataFrame(test_results)
df_results.to_csv(f'{type_path}_predictions.csv', index=False)

print("Prediccions guardades a predictions.csv")
