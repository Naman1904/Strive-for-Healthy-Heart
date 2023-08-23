# importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# loading and reading the dataset

heart = pd.read_csv("heart_cleveland_upload.csv")

# creating a copy of dataset so that will not affect our original dataset.
heart_df = heart.copy()

# Renaming some of the columns 
heart_df = heart_df.rename(columns={'condition':'target'})
print(heart_df.head())

# model building 

#fixing our data in x and y. Here y contains target data and X contains rest all the features.
x= heart_df.drop(columns= 'target')
y= heart_df.target

# splitting our dataset into training and testing for this we will use train_test_split library.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

#feature scaling
scaler= StandardScaler()
x_train_scaler= scaler.fit_transform(x_train)
x_test_scaler= scaler.fit_transform(x_test)

# creating randomForest classifier
model=RandomForestClassifier(n_estimators=20,random_state= 42)
model.fit(x_train_scaler, y_train)
y_pred= model.predict(x_test_scaler)
p = model.score(x_test_scaler,y_test)
print(p)

print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

cm = confusion_matrix(y_test, y_pred)
print(cm)
scores = cross_val_score(model,x,y,cv = 5)
mean_accuracy = scores.mean()
print('Mean Accuracy :{:.2f}%'.format(mean_accuracy*100)) 



model_xgb = xgb.XGBClassifier(n_estimators=20,random_state= 45)
model_xgb.fit(x_train_scaler, y_train)
y_pred_xgb = model_xgb.predict(x_test_scaler)
p_xgb = model_xgb.score(x_test_scaler,y_test)
print('XGBoost Classifier Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred_xgb)*100),2)))

print('XGBoost Classifier Classification Report\n', classification_report(y_test, y_pred_xgb))
print('XGBoost Classifier Confusion Matrix\n', confusion_matrix(y_test, y_pred_xgb))
scores_xgb = cross_val_score(model_xgb,x,y,cv = 5)
mean_accuracy_xgb = scores_xgb.mean()
print('XGBoost Classifier Mean Accuracy :{:.2f}%\n'.format(mean_accuracy_xgb*100))

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-knn-model.pkl'
pickle.dump(model, open(filename, 'wb'))