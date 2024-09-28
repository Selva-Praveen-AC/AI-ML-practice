from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
x,y=load_iris(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)

clf=KMeans(n_clusters=3,random_state=0).fit(x_train)
p_predict=clf.predict(x_test)
print(p_predict)

plt.scatter(x_test[:,0], x_test[:, 1], c=p_predict, cmap='viridis')
plt.show()
