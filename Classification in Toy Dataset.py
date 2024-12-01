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

#Output:
                  is cancer  is healthy
predicted cancer         40           8
predicted health          0          66 

              precision    recall  f1-score   support

           0       1.00      0.83      0.91        48
           1       0.89      1.00      0.94        66

    accuracy                           0.93       114
   macro avg       0.95      0.92      0.93       114
weighted avg       0.94      0.93      0.93       114

                  is cancer  is healthy
predicted cancer         44           4
predicted health          1          65 

              precision    recall  f1-score   support

           0       0.98      0.92      0.95        48
           1       0.94      0.98      0.96        66

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

                  is cancer  is healthy
predicted cancer         42           6
predicted health          1          65 

              precision    recall  f1-score   support

           0       0.98      0.88      0.92        48
           1       0.92      0.98      0.95        66

    accuracy                           0.94       114
   macro avg       0.95      0.93      0.94       114
weighted avg       0.94      0.94      0.94       114

                  is cancer  is healthy
predicted cancer         44           4
predicted health          0          66 

              precision    recall  f1-score   support

           0       1.00      0.92      0.96        48
           1       0.94      1.00      0.97        66

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114

                  is cancer  is healthy
predicted cancer         44           4
predicted health          2          64 

              precision    recall  f1-score   support

           0       0.96      0.92      0.94        48
           1       0.94      0.97      0.96        66

    accuracy                           0.95       114
   macro avg       0.95      0.94      0.95       114
weighted avg       0.95      0.95      0.95       114

Total value predicted : 114 Predicted labeled value : 108
                  is cancer  is healthy
predicted cancer         45           3
predicted health          0          66 

              precision    recall  f1-score   support

           0       1.00      0.94      0.97        48
           1       0.96      1.00      0.98        66

    accuracy                           0.97       114
   macro avg       0.98      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

Total value predicted : 114 Predicted labeled value : 111
                  is cancer  is healthy
predicted cancer         43           5
predicted health          2          64 

              precision    recall  f1-score   support

           0       0.96      0.90      0.92        48
           1       0.93      0.97      0.95        66

    accuracy                           0.94       114
   macro avg       0.94      0.93      0.94       114
weighted avg       0.94      0.94      0.94       114

Total value predicted : 114 Predicted labeled value : 107 
