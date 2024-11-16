import pandas as pd
svm_prediction = pd.read_csv('svm_prediction.csv')
rf_prediction = pd.read_csv('rf_prediction.csv')

# Merge the two predictions
for index, row in svm_prediction.iterrows():
    if row['des_value'] == 'INVALID':
        row['des_value'] = rf_prediction.loc[index, 'des_value']
    
svm_prediction.to_csv('rf_u_svm.csv', index=False)
        