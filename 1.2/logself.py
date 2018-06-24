import numpy as np
import pandas as pd

from pandas import Series, DataFrame
import scipy
from scipy.stats import spearmanr

from pylab import rcParams
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')

#reading the data
cars=pd.read_csv('mtcars.csv')

print(cars.shape)

cars.columns= ['car_name','mpg','cyl','disp','hp','drat','wt','qsec','vs','as','gear','carb']

print(cars.head())

#taking two values as
cars_data=cars.ix[:,(5,11)].values
cars_data_names=['drat','carb']

y=cars.ix[:,9].values
sb.regplot(x='drat',y='carb',data=cars,scatter=True)
plt.show()

drat=cars['drat']
carb=cars['carb']
spearmanr_coefficient, p_value=spearmanr(drat,carb)
print('Spearmanr Rank Correlation Coefficient %0.3f' % (spearmanr_coefficient))

#checking for missing values
cars.isnull().sum()
#checking id target is binary or ordinal
sb.countplot(x='as',data=cars,palette='hls')

#cheking if dat a is sufficient
cars.info()

x=scale(cars_data)

logreg=LogisticRegression()

logreg.fit(x, y)
#checking eficiency of model if score is one then model ifs perfect,if less than 0 then horrible
print(logreg.score(x, y))

y_pred=logreg.predict(x)
from sklearn.metrics import classification_report

print(classification_report(y, y_pred))

#data visualization
plt.plot(x, y,'ro')
plt.plot(x, y_pred)
plt.show()

