# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Create test set if you like to do the split programmatically or if you have not already split the data at this point

dataset = pd.read_csv('disadvantaged_communities.csv')

X = dataset.iloc[:, :-1] # : means select all rows/column      :-1 means all except the last
y = dataset.iloc[:, -1] # -1 means just the last column

df_cleaned = dataset.copy()

columns_with_missing = ['Linguistic Isolation','Linguistic Isolation Pctl', 'Unemployment','Unemployment Pctl', 'Lead','Lead Pctl', 'Traffic', 'Traffic Pctl', 'Low Birth Weight','Low Birth Weight Pctl', 'Education', 'Education Pctl','Housing Burden', 'Housing Burden Pctl']
for col in columns_with_missing:
    df_cleaned.dropna(subset=[col],how='any',inplace=True)

encoder = ce.OrdinalEncoder(cols=['DAC category', 'CES 4.0 Percentile Range', 'California County', 'Approximate Location'])
df_encoded = encoder.fit_transform(df_cleaned)

df_cleaned.isnull().sum()


X = df_encoded.drop(['CES 4.0 Percentile Range'], axis=1)
y = df_encoded['CES 4.0 Percentile Range']


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.10, random_state = 0)


# Standardize the features

cols = X_train.columns
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Below is the code to convert X_train and X_test into data frames for the next steps

X_train = pd.DataFrame(X_train, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test, columns=[cols]) 

# 3. Load your saved model for dissadvantaged communities classification 
#that you saved in dissadvantaged_communities_classification.py via Pikcle

svm = 'SvmClassifier.sav'
svm_model = pickle.load(open(svm, 'rb'))


rf = 'RfClassifier.sav'
rf_model = pickle.load(open(rf, 'rb'))


#Print out for the Support Vector Machine


# 4. Make predictions on test_set created from step 2


# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.

y_pred = svm_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))
# Get and print confusion matrix

#cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# Compute Classification Accuracy and use the following line to print it
classification_accuracy = (TP + TN)/(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Compute Precision and use the following line to print it
precision = TP / (TP + FP) # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
recall = TP / (TP + FN) # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN + FP) # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))






#Print out for the Random Forest Variable



y_pred = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))
# Get and print confusion matrix

#cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# Compute Classification Accuracy and use the following line to print it
classification_accuracy = (TP + TN)/(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Compute Precision and use the following line to print it
precision = TP / (TP + FP) # Change this line to implement Precision formula
print('Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
recall = TP / (TP + FN) # Change this line to implement Recall formula
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN + FP) # Change this line to implement Specificity formula
print('Specificity : {0:0.3f}'.format(specificity))

