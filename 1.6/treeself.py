import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#getting data from internet
data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data")

#setting up data
x=data.iloc[:,[1,2,3,4]]
y=data.iloc[:,[0]]

#data preprocessiong

le=preprocessing.LabelEncoder()
y=le.fit_transform(y)

#Lis 1 R is 2 and B is 0


#splitting data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

#training classifer
#tuning classifier
clf=tree.DecisionTreeClassifier(criterion='gini',min_samples_split=30,splitter='best')

clf=clf.fit(x_train,y_train)

#prediction

y_pred=clf.predict(x_test)

#testing accuracy
accuracy=accuracy_score(y_test,y_pred)

print(str(accuracy*100)+"% accuracy")

#visualizing data
#we can't scatter since ther should be two variables only
height=pd.Series(y).value_counts(normalize=True)
plt.bar(range(3),height.tolist()[::-1],1/1.5,color='red',label="Classes",alpha=0.8)#alpha is opacity
plt.title('Dcision Tree Classification')
plt.xlabel('B L R')
plt.ylabel('Occurences')
plt.legend()
plt.show()

