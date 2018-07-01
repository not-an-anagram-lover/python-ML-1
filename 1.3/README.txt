A Support Vector Machine (SVM) is a supervised machine learning algorithm that can be employed for both classification and regression purposes.
 SVMs are more commonly used in classification problems.

SVMs are based on the idea of finding a hyperplane that best divides a dataset into two classes
you can think of a hyperplane as a line that linearly separates and classifies a set of data.

 In order to classify a dataset it’s necessary to move away from a 2d view of the data to a 3d view. Explaining this is easiest with another simplified example.
 Imagine that our two sets of colored balls above are sitting on a sheet and this sheet is lifted suddenly, launching the balls into the air. 
While the balls are up in the air, you use the sheet to separate them. 
This ‘lifting’ of the balls represents the mapping of data into a higher dimension. This is known as kernelling.

#Import Library
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.svc(kernel='linear', c=1, gamma=1) 
# there is various option associated with it, like changing kernel, 
#gamma and C value.
#Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)