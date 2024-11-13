from keras import models
from keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer=load_iris()

x_data=cancer.data
y_data=cancer.target

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=20)

model=models.Sequential()

model.add(layers.Dense(10,activation='relu',input_shape=[x_train.shape[1]]))
model.add(layers.Dense(10,activation='relu'))
model.add(layers.Dense(1))
print(model.summary())
print(model.get_config())

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mse'])

model.fit(x_train,y_train,batch_size=50,validation_split=0.2,epochs=50,verbose=1)

result=model.evaluate(x_test,y_test)
print(f"lose - {result[0]}")
print(f"Accuracy : {result[1]}")

plt.scatter(result[0],result[1],c='y')
plt.show()
