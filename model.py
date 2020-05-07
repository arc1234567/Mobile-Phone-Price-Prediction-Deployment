import pandas as pd
import numpy as np
df=pd.read_csv("data_set.csv")
x=df.drop(['Price'],axis=1)
y=df['Price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)

model.score(x_test,y_test)

import pickle
# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model1 = pickle.load(open('model.pkl','rb'))

model1.predict([[4.2,2000,32]])