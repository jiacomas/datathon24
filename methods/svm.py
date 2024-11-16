import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier

file = '/Users/jia/Documents/datathon24/clean_data.csv'
labels = [
    'silhouette_type', 'neck_lapel_type', 'woven_structure', 'knit_structure',
    'heel_shape_type', 'length_type', 'sleeve_length_type', 'toecap_type',
    'waist_type', 'closure_placement', 'cane_height_type'
]

def get_numeric_data(file, categorical_columns, encoder=LabelEncoder()):
    bodegon = pd.read_csv(file)
    bodegon[categorical_columns] = bodegon[categorical_columns].fillna("INVALID")
    for column in categorical_columns:
        bodegon[column] = bodegon[column].astype(str)
    for column in categorical_columns:
        bodegon[column] = encoder.fit_transform(bodegon[column])
    return bodegon

categories_data = [
    'des_color', 'des_sex', 'des_age', 'des_line', 'des_fabric', 
    'des_product_category', 'des_product_aggregated_family', 'des_product_family', 
    'des_product_type', 'silhouette_type', 'neck_lapel_type', 'woven_structure',
    'knit_structure', 'heel_shape_type', 'length_type', 'sleeve_length_type',
    'toecap_type', 'waist_type', 'closure_placement', 'cane_height_type'
]

categories_test = [
    "cod_color", "des_sex", "des_age", "des_line", "des_fabric", 
    "des_product_category", "des_product_aggregated_family", 
    "des_product_family", "des_product_type", "des_filename", "des_color"
]

data = get_numeric_data(file, categories_data)

encoding = pd.read_csv("/Users/jia/Documents/datathon24/archive/attribute_data.csv")
encoding_dict = {}

for _, row in encoding.iterrows():
    attribute_name = row['attribute_name']
    cod_value = row['cod_value']
    des_value = row['des_value']

    if attribute_name not in encoding_dict:
        encoding_dict[attribute_name] = {}

    encoding_dict[attribute_name][cod_value] = des_value

# Prepare the data
X = data.drop(labels + ["cod_modelo_color", "des_filename", "cod_color"], axis=1)
y = data[labels]

# Split and normalize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = Normalizer().fit(X_train)
normalized_x_train = scaler.transform(X_train)
normalized_x_test = scaler.transform(X_test)

# Use MultiOutputClassifier
svm = SVC(kernel='linear', C=1.0)
multi_output_model = MultiOutputClassifier(svm)
multi_output_model.fit(normalized_x_train, y_train)

# Calculate train accuracy
y_train_pred = multi_output_model.predict(normalized_x_train)
train_accuracy = [accuracy_score(y_train.iloc[:, i], y_train_pred[:, i]) for i in range(len(labels))]
average_train_accuracy = sum(train_accuracy) / len(train_accuracy)
print(f"Average train accuracy: {average_train_accuracy:.2f}")

# Initialize a DataFrame to store all predictions
all_predictions = pd.DataFrame(columns=['test_id', 'label', 'des_value'])

# Iterate through each label to make and save predictions
for i, label in enumerate(labels):
    # Load test data
    test_data_file = f'/Users/jia/Documents/datathon24/divide/{label}_data.csv'
    test_data = get_numeric_data(test_data_file, categories_test)
    X_test_data = test_data.drop(
        ['attribute_name', 'cod_color', 'cod_modelo_color', 'des_filename', 'test_id'], axis=1
    )
    X_test_data = X_test_data[[col for col in X_test_data.columns if col != X_test_data.columns[0]] + [X_test_data.columns[0]]]
    normalized_test_data = scaler.transform(X_test_data)
    
    # Make predictions for the current label
    predictions = multi_output_model.estimators_[i].predict(normalized_test_data)
    
    # Map predictions to descriptive values
    mapped_predictions = [
        encoding_dict[label].get(int(pred), "INVALID") for pred in predictions
    ]
    
    # Collect predictions in a DataFrame
    temp_df = pd.DataFrame({
        'test_id': test_data['test_id'],
        'des_value': mapped_predictions
    })
    all_predictions = pd.concat([all_predictions, temp_df], ignore_index=True)

# Save all predictions in a single CSV file
all_predictions.to_csv('/Users/jia/Documents/datathon24/svm_predictions.csv', index=False)
print("All predictions saved successfully.")
