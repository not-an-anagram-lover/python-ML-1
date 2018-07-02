import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylab import rcParams
import seaborn as sb

from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from pandas_ml import ConfusionMatrix
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


class Linear:
    def __init__(self):
        self.df=pd.read_csv('linear.csv')
        self.df.columns=['X','Y']
        self.df.head()

        self.linear =LinearRegression()
        self.trainX = np.asarray(self.df.X[20:len(self.df.X)]).reshape(-1, 1)
        self.trainY = np.asarray(self.df.Y[20:len(self.df.Y)]).reshape(-1, 1)
        self.testX = np.asarray(self.df.X[:20]).reshape(-1, 1)
        self.testY = np.asarray(self.df.Y[:20]).reshape(-1, 1)
        self.linear.fit(self.trainX, self.trainY)

        self.show()
        self.check()


    def show(self):
        sb.set_context("notebook",font_scale=1.1)
        sb.set_style("ticks")
        sb.lmplot('X','Y',data=self.df)
        plt.ylabel('Response')
        plt.xlabel('Explantory')
        plt.show()

    def check(self):
        print("Accuracy Linear:\n",self.linear.score(self.trainX, self.trainY))
        print('Coefficient:\n',self.linear.coef_)
        print('Intercept:\n',self.linear.intercept_)
        print('R2 Value:\n',self.linear.score(self.trainX,self.trainY))
        self.predictions=self.linear.predict(self.testX)

class Logistic:
    def __init__(self):
        self.df=pd.read_csv('logistic.csv')
        self.df.columns=['X','Y']
        self.df.head()
        self.logistic = LogisticRegression()
        self.X=(np.asarray(self.df.X)).reshape(-1,1)
        self.Y=(np.asarray(self.df.Y)).ravel()
        self.logistic.fit(self.X,self.Y)


        self.show()
        self.check()
    def show(self):
        sb.set_context("notebook", font_scale=1.1)
        sb.set_style("ticks")
        sb.regplot('X', 'Y', data=self.df,logistic=True)
        plt.ylabel('Probability')
        plt.xlabel('Explantory')
        plt.show()

    def check(self):
        print('Accuracy Logistic:\n', self.logistic.score(self.X, self.Y))
        print('Coefficient:\n', self.logistic.coef_)
        print('Intercept:\n', self.logistic.intercept_)
        print('R2 Value:\n', self.logistic.score(self.X, self.Y))
        self.predictions = self.logistic.predict(self.X)





class Dtree:
    def __init__(self):
        self.df=pd.read_csv('iris.data.csv')
        self.df.columns=['X1','X2','X3','X4','Y']
        self.df.head()

        self.decision=tree.DecisionTreeClassifier(criterion='gini')
        self.X=self.df.values[:,0:4]
        self.Y=self.df.values[:,4]
        self.trainX,self.testX,self.trainY,self.testY=train_test_split(self.X,self.Y,test_size=0.3)
        self.decision.fit(self.trainX,self.trainY)
        self.check()
        self.show()
    def check(self):
        print('Accuracy Dtree:\n', self.decision.score(self.testX, self.testY))

        self.predictions = self.decision.predict(self.testX)

        print(metrics.classification_report(self.testY, self.predictions))
        print(metrics.confusion_matrix(self.testY, self.predictions))

    def show(self):

        plt.plot(self.testX,self.predictions,'ro')
        plt.xlabel('Sample_test_cases')
        plt.ylabel('Target_predicted cases')
        plt.show()





class Kmean:
    def __init__(self):
        self.iris = datasets.load_iris()

        self.X = self.iris.data[:, 1:3]
        self.Y=self.iris.target
        self.model = KMeans(n_clusters=3, random_state=0)
        self.model.fit(self.X)
        self.show()
        self.check()
    def show(self):
        self.centroids = self.model.cluster_centers_

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='^', s=170, zorder=10, c='m')
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.model.labels_)
        plt.xlabel("Sepal width")
        plt.ylabel("Petal length")
        plt.show()

    def check(self):
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.Y, test_size=0.3)
        self.model.fit(self.trainX, self.trainY)
        print('Accuracy Kmeans:\n', self.model.score(self.testX, self.testY))
        self.predictions = self.model.predict(self.testX)

        print(metrics.classification_report(self.testY, self.predictions))
        print(metrics.confusion_matrix(self.testY, self.predictions))


class Knn:
    def __init__(self):
        self.iris = datasets.load_iris()

        self.X = self.iris.data
        self.y = self.iris.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15, random_state=2)

        self.model = neighbors.KNeighborsClassifier(n_neighbors=50)
        self.model.fit(self.X_train, self.y_train)

        self.show()
        self.check()

    def show(self):
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train)
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c='m')
        plt.xlabel("Sepal length")
        plt.ylabel("Sepal width")
        plt.show()

    def check(self):
        print('Accuracy Knn:\n',self.model.score(self.X_test, self.y_test))

        self.predictions = self.model.predict(self.X_test)

        print(metrics.classification_report(self.y_test, self.predictions))
        print(metrics.confusion_matrix(self.y_test, self.predictions))

class Svm:
    def __init__(self):
        self.iris = datasets.load_iris()

        self.X = self.iris.data[:,:2]
        self.y = self.iris.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.15, random_state=2)

        self.model = SVC(kernel='poly')
        self.model.fit(self.X_train, self.y_train)

        self.show()
        self.check()

    def show(self):
        h = .02  # step size in mesh
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])

        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.ocean)
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def check(self):
        print('Accuracy SVM:\n', self.model.score(self.X_test, self.y_test))

        self.predictions = self.model.predict(self.X_test)

        print(metrics.classification_report(self.y_test, self.predictions))
        print(metrics.confusion_matrix(self.y_test, self.predictions))


class Baeyes:
    def __init__(self):
        self.df = pd.read_csv('iris.data.csv')
        self.X =self.df.iloc[:, :4].values
        self.y = self.df.iloc[:,4].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=82)
        self.nv=GaussianNB()
        self.nv.fit(self.X_train,self.y_train)


        self.check()
        self.show()



    def show(self):

        plt.plot(self.X_test,self.predictions,'ro')
        plt.xlabel('Sample_test_cases')
        plt.ylabel('Target_predicted cases')
        plt.show()

    def check(self):
        print('Accuracy Baeyes:\n', self.nv.score(self.X_test, self.y_test))

        self.predictions = self.nv.predict(self.X_test)
        self.com=np.vstack((self.y_test,self.predictions))
        print(self.com[:5,:])

        print(metrics.classification_report(self.y_test, self.predictions))
        print(metrics.confusion_matrix(self.y_test, self.predictions))



print('1.Linear Regression')
print('2.Logistic Regression')
print('3.SVM')
print('4.KNN')
print('5.Baeyes')
print('6.K-Means')
print('7.Decision-Tree')
print('8.Compare')
print('9.Exit')
ch=int(input("enter choice"))
c=ch
while((c>0)&(c<10)):
    if (c == 1):
        ll = Linear()
    elif (c == 2):
        lh = Logistic()
    elif (c == 3):
        le = Svm()
    elif (c == 4):
        lo = Knn()
    elif (c == 5):
        lp = Baeyes()
    elif (c == 6):
        li = Kmean()
    elif(c == 7):
        lin=Dtree()

    else:
        print('wrong choice')
    c=int(input('enter 9 to exit'))
