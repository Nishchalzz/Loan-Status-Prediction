import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm


loan_dataset = pd.read_csv('contents/loan_dataset.csv')
loan_dataset = loan_dataset.dropna()
loan_dataset.replace({"Loan_Status": {'Y': 1, 'N': 0}}, inplace=True)
loan_dataset.replace({'Dependents': {'3+': 4}}, inplace=True)
loan_dataset.replace({
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Self_Employed': {'No': 0, 'Yes': 1}
}, inplace=True)

X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.1, random_state=2)


classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)


input_data = [float(i) for i in sys.argv[1:]]
input_data_as_nparray = np.asarray(input_data)
input_data_reshaped = input_data_as_nparray.reshape(1, -1)
result = classifier.predict(input_data_reshaped)


if result[0] == 1:
    print("Approved")
else:
    print("Not Approved")
