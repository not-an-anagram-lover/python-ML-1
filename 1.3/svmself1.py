import numpy as np
from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

def evaluate_data(model=None):
    pred=model.predict(x_test)
    rclass=0
    for i in range (len(y_test)):
        if pred[i]==y_test[i]:
            rclass+=1
    accuracy=100*rclass/len(y_test)
    return accuracy

#rbf kernel is also called gaussian
kernels=('linear','poly','rbf')
accuracies=[]
#accuracies of differtnt kernels
for index, kernel in enumerate(kernels):
    model=svm.SVC(kernel=kernel)
    model.fit(x_train,y_train)
    acc=evaluate_data(model)
    accuracies.append(acc)
    print("{} % accuracy obtained with kernel={}".format(acc,kernel))

#train SVM's with different kernels

svc=svm.SVC(kernel='linear').fit(x_train,y_train)
rbf_svc=svm.SVC(kernel='rbf',gamma=0.7).fit(x_train,y_train)
poly_svc=svm.SVC(kernel='poly',degree=3).fit(x_train,y_train)

#create a mesh to plot in
h=.02#step size in mesh
x_min, x_max=x[:,0].min()-1,x[:,0].max()+1
y_min, y_max=x[:,1].min()-1,x[:,1].max()+1
xx, yy=np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min,y_max, h))

#define titles for the plots
titles=['SVC with linear kernel',
        'SVC with RBF kernel',
        'SVC with polynomial(degree 3) kernel']
for i,clf in enumerate((svc,rbf_svc,poly_svc)):
    #plot decision boundary
    #poitn in the mesh[x_min,m_max[y_min,y_max]
    plt.figure(i)

    z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
# put result into a color plot
    z=z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(x[:,0],x[:,1], c=y, cmap=plt.cm.ocean)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()