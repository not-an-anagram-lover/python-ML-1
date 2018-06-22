import numpy as py
import pandas as pd
import matplotlib.pyplot as plt

#reading the file

datasets = pd.read_csv('Salary.csv')

#dividing dataset into x and y

X=datasets.iloc[:,:-1].values
Y=datasets.iloc[:,1].values
print(X)
print(Y)

#splitiing dataset into test and train

from sklearn.cross_validation import  train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
print('independent train')
print(X_train)
print('dependent train')
print(Y_train)


print('independent test')
print(X_test)
print('dependent test')
print(Y_test)


#implement our classifier

from sklearn.linear_model import LinearRegression

linreg=LinearRegression()

linreg.fit(X_train,Y_train)

y_predict=linreg.predict(X_test)

d={'org':Y_test,'pre':y_predict}
df=pd.DataFrame(data=d)
print(df)


#Implement the graphs

plt.scatter(X_train,Y_train)
plt.plot(X_train,linreg.predict(X_train),color='red')
plt.show()


linreg.fit(X_test,Y_test)
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,linreg.predict(X_test))
plt.show()