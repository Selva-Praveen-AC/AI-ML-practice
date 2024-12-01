from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score
import pandas as pd

iris = load_iris()

iris_pf=pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_pf['species']=iris.target

print(iris_pf.describe(),"\n")

x,y=load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=20)

clf=KNeighborsClassifier()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)

print(f"Accuracy point : \n{accuracy_score(y_test,y_predict)}\n")
cm=confusion_matrix(y_test,y_predict)
classes=iris.target_names
frame=pd.DataFrame(cm,index=[f'predicted {cls}'for cls in classes],columns=[f'is {cls}' for cls in classes])
print(f"Confusion matrix point : \n{frame}\n")
print(f"Classification Report :\n {classification_report(y_test,y_predict)}")
print(f"Precision Score : \n{precision_score(y_test,y_predict,average='micro')}")

#Output:
       sepal length (cm)  sepal width (cm)  petal length (cm)  \
count         150.000000        150.000000         150.000000   
mean            5.843333          3.057333           3.758000   
std             0.828066          0.435866           1.765298   
min             4.300000          2.000000           1.000000   
25%             5.100000          2.800000           1.600000   
50%             5.800000          3.000000           4.350000   
75%             6.400000          3.300000           5.100000   
max             7.900000          4.400000           6.900000   

       petal width (cm)     species  
count        150.000000  150.000000  
mean           1.199333    1.000000  
std            0.762238    0.819232  
min            0.100000    0.000000  
25%            0.300000    0.000000  
50%            1.300000    1.000000  
75%            1.800000    2.000000  
max            2.500000    2.000000   

Accuracy point : 
0.9666666666666667

Confusion matrix point : 
                      is setosa  is versicolor  is virginica
predicted setosa              8              0             0
predicted versicolor          0             10             1
predicted virginica           0              0            11

Classification Report :
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         8
           1       1.00      0.91      0.95        11
           2       0.92      1.00      0.96        11

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30

Precision Score : 
0.9666666666666667
