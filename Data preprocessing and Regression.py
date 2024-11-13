import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_absolute_error,r2_score


data = {
    'Age': [25, np.nan, 35, 45, 32, np.nan, 40],
    'Salary': [50000, 54000, np.nan, 62000, 58000, 60000, np.nan],
    'Gender': ['Male', 'Female', 'Female', np.nan, 'Female', 'Male', 'Male'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df=pd.DataFrame(data)

impute=SimpleImputer(strategy='mean')

df[['Salary','Age']]=impute.fit_transform(df[['Salary','Age']])

gender_impute=SimpleImputer(strategy='most_frequent')
df[['Gender']]=gender_impute.fit_transform(df[['Gender']])

encode=LabelEncoder()
df['Gender']=encode.fit_transform(df['Gender'])

x=df[['Gender','Age']]
y=df['Salary']

names=["Super vector regression","Logistic Regression"]
classifications=[SVR(),LogisticRegression(max_iter=1000)]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
for name,clf in zip(names,classifications):
    clf.fit(x_train,y_train)
    c_predict=clf.predict(x_test)
 
    print(f"{name} - Mean Absolute Error : {mean_absolute_error(y_test,c_predict)}")
    print(f"{name} - R2 Score : {r2_score(y_test,c_predict)}")
