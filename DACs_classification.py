#Step 1:
# Import libraries
# In this section, you can use a search engine to look for the functions that will help you implement the following steps

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Step 2:
# Load dataset and show basic statistics
# 1. Show dataset size (dimensions)
# 2. Show what column names exist for the 49 attributes in the dataset
# 3. Show the distribution of the target class CES 4.0 Percentile Range column
# 4. Show the percentage distribution of the target class CES 4.0 Percentile Range column

dataset = pd.read_csv('disadvantaged_communities.csv')

print("Data Shape:", dataset.shape)
print("Column Names:", dataset.columns)
print("Target Class Distribution:\n", dataset['CES 4.0 Percentile Range'].value_counts())
print("Percentage Distribution of Target Class:\n", dataset['CES 4.0 Percentile Range'].value_counts(normalize=True))
print("Missing Values Summary:\n", dataset.isnull().sum())
print("Percentage of Missing Values in Data:", (dataset.isnull().sum().sum() / np.product(dataset.shape)) * 100)

X = dataset.iloc[:, :-1] # : means select all rows/column      :-1 means all except the last
y = dataset.iloc[:, -1] # -1 means just the last column

# Step 3:
#Clean the dataset - you can eitherhandle the missing values in the dataset
# with the mean of the columns attributes or remove rows the have missing values.

#cleandata = dataset.copy()
#emptyCells = ['Linguistic Isolation', 'Linguistic Isolation Pctl', 'Unemployment', 'Unemployment Pctl', 'Lead', 'Lead Pctl', 'Traffic', 'Traffic Pctl', 'Low Birth Weight', 'Low Birth Weight Pctl', 'Housing Burden', 'Housing Burden Pctl']

#for col in emptyCells:
#   mValue = cleandata[col].mean() #Find mean value
#   cleandata.fillna(mValue, inplace = True)

df_cleaned = dataset.copy()

columns_with_missing = ['Linguistic Isolation','Linguistic Isolation Pctl', 'Unemployment','Unemployment Pctl', 'Lead','Lead Pctl', 'Traffic', 'Traffic Pctl', 'Low Birth Weight','Low Birth Weight Pctl', 'Education', 'Education Pctl','Housing Burden', 'Housing Burden Pctl']
for col in columns_with_missing:
    df_cleaned.dropna(subset=[col],how='any',inplace=True)

df_cleaned.isnull().sum()


# Step 4: 
#Encode the Categorical Variables - Using OrdinalEncoder from the category_encoders library to encode categorical variables as ordinal integers
import category_encoders as ce


encoder = ce.OrdinalEncoder(cols=['DAC category', 'CES 4.0 Percentile Range', 'California County', 'Approximate Location'])
df_encoded = encoder.fit_transform(df_cleaned)

#enc = ce.OrdinalEncoder(col = ['California County', 'Approximate Location', 'CES 4.0 Percintile Range', 'DAC category'])

#encdataset = enc.fit_transform(dataset)

#encdata = cleandata

#catcol = [col for col in encdata.columns if encdata[col].dtype == 'object']

#enc = ce.OrdinalEncoder(col = catcol)

#dataset = enc.fit_transform(dataset)


# Step 5: 
# Separate predictor variables from the target variable (attributes (X) and target variable (y) as we did in the class)
# Create train and test splits for model development. Use the 80% and 20% split ratio
# Use stratifying (stratify=y) to ensure class balance in train/test splits
# Name them as X_train, X_test, y_train, and y_test
# Name them as X_train, X_test, y_train, and y_test

from sklearn.model_selection import train_test_split


#X = df_encoded.iloc[:, :-1] # : means select all rows/column      :-1 means all except the last
#y = df_encoded.iloc[:, -1] # -1 means just the last column
X = df_encoded.drop(['CES 4.0 Percentile Range'], axis=1)
y = df_encoded['CES 4.0 Percentile Range']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.20, random_state = 0)


#X_train = [] # Remove this line after implementing train test split
#X_test = [] # Remove this line after implementing train test split


# Do not do steps 6 - 8 for the Ramdom Forest Model
# Step 6:
# Standardize the features (Import StandardScaler here)
from sklearn.preprocessing import StandardScaler

cols = X_train.columns
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Step 7:
# Below is the code to convert X_train and X_test into data frames for the next steps

X_train = pd.DataFrame(X_train, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test, columns=[cols]) # pd is the imported pandas lirary - Import pandas as pd



# Step 8 - Build and train the SVM classifier
# Train SVM with the following parameters. (use the parameters with the highest accuracy for the model)
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3 (Linear)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', C = 10.0, gamma = 0.3)
classifier.fit(X_train, y_train)

# Test the above developed SVC on unseen pulsar dataset samples

# compute and print accuracy score

#print(classifier.predict(sc.transform([[30, 87000]])))

from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))

# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
import pickle

filename = 'SvmClassifier.sav'
pickle.dump(classifier, open(filename, 'wb'))



# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix

#cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity

#Results are in evaluation.py file

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






# Step 9: Build and train the Random Forest classifier
# Train Random Forest  with the following parameters.
# (n_estimators=10, random_state=0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rfclassifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
rfclassifier.fit(X_train, y_train)

# Test the above developed Random Forest model on unseen DACs dataset samples

# compute and print accuracy score

#print(rfclassifier.predict(sc.transform([[30, 87000]])))

from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = rfclassifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))

# Save your Random Forest model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment

filename = 'RfClassifier.sav'
pickle.dump(rfclassifier, open(filename, 'wb'))

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
#cm = [[]]
# Below are the metrics for computing classification accuracy, precision, recall and specificity

#Results are in evaluation.py file

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
