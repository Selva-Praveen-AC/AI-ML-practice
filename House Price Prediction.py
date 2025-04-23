import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression
house_data = {
    "House_ID": ["H001", "H002", "H003", "H004", "H005"],
    "Location": ["New York", "Los Angeles", "Chicago", None, "San Francisco"],
    "Size_in_sqft": [1500, None, 1300, 1600, 1800],
    "Bedrooms": [3, 4, 3, 3, None],
    "Bathrooms": [2, 3, None, 2, 2],
    "Year_Built": [2005, 2010, 2000, None, 2015],
    "Price": [750000, 820000, None, 690000, 950000]
}

models={
    "Linear Regression":LinearRegression(),
}

df = pd.DataFrame(house_data)

df=df.replace({None:np.nan})
impute= SimpleImputer(strategy='most_frequent')
df['Location']=impute.fit_transform(df[['Location']]).ravel()

impute=SimpleImputer(strategy='mean')
col=['Size_in_sqft','Bedrooms','Bathrooms','Year_Built','Price']
df[col]=impute.fit_transform(df[col])
print(df[col])


encode=LabelEncoder()
df['Location']=encode.fit_transform(df['Location'])
df['Location']

x=df.drop(columns=['House_ID','Price'])
y=df['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20)


for mdl,clf in models.items():
    
    model=clf.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    score=clf.score(x_test,y_test)
    print(f"Predicted Prices: {y_pred}")
    print(f"{mdl} : {score}")

