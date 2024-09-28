#Classification in Toy Datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

cancer=load_breast_cancer()

x=cancer.data
y=cancer.target

#Training the Data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)

# SVM Classification
clf=SVC()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

cm=confusion_matrix(y_test,y_predict)
frame=pd.DataFrame(cm,index=['predicted cancer','predicted health'],columns=['is cancer','is healthy'])
print(frame,"\n")
print(classification_report(y_test,y_predict))

# Naive Bayes Classification
clf=GaussianNB()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

cm=confusion_matrix(y_test,y_predict)
frame=pd.DataFrame(cm,index=['predicted cancer','predicted health'],columns=['is cancer','is healthy'])
print(frame,"\n")
print(classification_report(y_test,y_predict))

#Principal Discriminant Analysis
clf=PCA()
clf.fit(x_train,y_train)
x_reduced_test=clf.transform(x_test)
x_reduced_train=clf.transform(x_train)

clf=SVC()
clf.fit(x_reduced_train,y_train)
y_predict=clf.predict(x_reduced_test)

cm=confusion_matrix(y_test,y_predict)
frame=pd.DataFrame(cm,index=['predicted cancer','predicted health'],columns=['is cancer','is healthy'])
print(frame,"\n")
print(classification_report(y_test,y_predict))

#Linear Discriminant Analysis
clf=LinearDiscriminantAnalysis()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

cm=confusion_matrix(y_test,y_predict)
frame=pd.DataFrame(cm,index=['predicted cancer','predicted health'],columns=['is cancer','is healthy'])
print(frame,"\n")
print(classification_report(y_test,y_predict))

#Decision Forest classification
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

cm=confusion_matrix(y_test,y_predict)
frame=pd.DataFrame(cm,index=['predicted cancer','predicted health'],columns=['is cancer','is healthy'])
print(frame,"\n")
print(classification_report(y_test,y_predict))

N=y_test.shape[0]
C=(y_test==y_predict).sum()
print(f"Total value predicted : {N} Predicted labeled value : {C}")

#Random Forest Classification
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

cm=confusion_matrix(y_test,y_predict)
frame=pd.DataFrame(cm,index=['predicted cancer','predicted health'],columns=['is cancer','is healthy'])
print(frame,"\n")
print(classification_report(y_test,y_predict))

N=y_test.shape[0]
C=(y_test==y_predict).sum()
print(f"Total value predicted : {N} Predicted labeled value : {C}")

#K-Nearest Neighbor Classification
clf=KNeighborsClassifier()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

cm=confusion_matrix(y_test,y_predict)
frame=pd.DataFrame(cm,index=['predicted cancer','predicted health'],columns=['is cancer','is healthy'])
print(frame,"\n")
print(classification_report(y_test,y_predict))

N=y_test.shape[0]
C=(y_test==y_predict).sum()
print(f"Total value predicted : {N} Predicted labeled value : {C}")
