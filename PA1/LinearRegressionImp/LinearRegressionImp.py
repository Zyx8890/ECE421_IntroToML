import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fit_LinRegr(X_train, y_train):
	#Add implementation here
	N = np.shape(X_train)[0]
	d = np.shape(X_train)[1]
	X = np.ones(shape=(N,d+1))
	X[:,0:-1] = X_train
	try:
		A = np.linalg.pinv(X.T @ X)
	except:
		raise ValueError("A singular")
	w = A @ X.T @ y_train
	return w

def mse(X_train,y_train,w):
	#Add implementation here
	N = np.shape(X_train)[0]
	d = np.shape(X_train)[1]
	X = np.ones(shape=(N,d+1))
	X[:,0:-1] = X_train
	h = w.T @ X.T
	return 1/N * sum((h - y_train)**2)


def pred(X_train,w):
	#Add implementation here
	return np.dot(w, X_train)

def test_SciKit(X_train, X_test, Y_train, Y_test):
	#Add implementation here
	# Train model
	LR = linear_model.LinearRegression()
	LR.fit(X_train, Y_train)
	# print(LR.coef_)
	return mean_squared_error(LR.predict(X_test), Y_test)

def subtestFn():
	# This function tests if your solution is robust against singular matrix

	# X_train has two perfectly correlated features
	X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
	y_train = np.asarray([1,2,3,4])
	
	try:
		w=fit_LinRegr(X_train, y_train)
		print ("weights: ", w)
		print ("NO ERROR")
	except:
		print ("ERROR")

def test_LR():
	X_train, y_train = load_diabetes(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
	
	w=fit_LinRegr(X_train, y_train)
	# print(w)
	e=mse(X_test,y_test,w)
	
	#Testing Part 2b
	scikit=test_SciKit(X_train, X_test, y_train, y_test)
	
	print("Mean squared error from Part 2a is ", e)
	print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
# testFn_Part2()
test_LR()

# The sklearn implementation is very close to the exact solution obtained from the linear regression
# Difference: 9.640643838793039e-11
# Percentage deviation: ~1e-12 %