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
