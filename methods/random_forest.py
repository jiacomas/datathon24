import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_numeric_data(file, encoder=LabelEncoder()):
    bodegon = pd.read_csv(file)
    categorical_columns = [
        'des_color', 'des_sex', 'des_age', 'des_line', 'des_fabric', 
        'des_product_category', 'des_product_aggregated_family', 'des_product_family', 
        'des_product_type', 'silhouette_type', 'neck_lapel_type', 'woven_structure',
        'knit_structure', 'heel_shape_type', 'length_type', 'sleeve_length_type',
        'toecap_type', 'waist_type', 'closure_placement', 'cane_height_type'
    ]
    
    bodegon[categorical_columns] = bodegon[categorical_columns].fillna("INVALID")
    for column in categorical_columns:
        bodegon[column] = bodegon[column].astype(str)
    bodegon[categorical_columns] = bodegon[categorical_columns].fillna(-1)
    for column in categorical_columns:
        bodegon[column] = encoder.fit_transform(bodegon[column])
    return bodegon

if __name__=="__main__":
    file = '/Users/luke/datathon24/bodegon.csv'
    labels = ['silhouette_type', 'neck_lapel_type', 'woven_structure','knit_structure', 'heel_shape_type', 'length_type','sleeve_length_type', 'toecap_type', 'waist_type','closure_placement', 'cane_height_type']

    data_clean = get_numeric_data(file)    
    sc=0
    for i in range(len(labels)):
        X = data_clean.drop(labels, axis=1)
        X = X.drop(["cod_modelo_color", "des_filename","cod_color"], axis=1)

        y = data_clean[labels[6]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X, y)
        scores = cross_val_score(model, X, y, cv=4) 
        sc+=scores.mean()

    print("Precision:",sc/len(labels))
    

    # param_grid = {
    #     'n_estimators': [100, 200, 500],
    #     'max_depth': [10, 20, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }

    # grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=2, scoring='accuracy')
    # grid_search.fit(X_train, y_train)

    # print("Mejores parámetros encontrados: ", grid_search.best_params_)
    # model = grid_search.best_estimator_