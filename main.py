import pandas
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

data= pandas.read_csv('lab3data.csv')
x = data['population'].values
y =data ['profit'].values
m = y.shape[0]
theta0 = 0
theta1 = 0
iterations = 1500
alpha = 0.01
def predict (x , theta):
    h = theta[0] +theta[1]*x
    return h



def calc_error (h ,y ):
    j=(1/m)*sum((h-y)**2)
    return j

def gradient_descent (x , y ,theta, alpha , num_iters):
    costs = np.zeros(shape = (num_iters))
    errors = np.zeros(shape = (num_iters))

    for i in range(num_iters):
        prdectionvar = predict(x, theta)
        cost = calc_error(prdectionvar, y)
        error = np.sum((prdectionvar-y)**2)
        diff_theta0 = (1/m)*sum((prdectionvar-y)*1)
        diff_theta1 = (1/m)*sum((prdectionvar-y)*x)
        theta[0] -= alpha * diff_theta0
        theta[1] -= alpha * diff_theta1
        errors[i] = error
        costs[i] = cost
    return theta, costs, errors
b = [theta0,theta1]

theta , costs , errors = gradient_descent(x,y,b,alpha,iterations)
print(theta)
plt.plot(list(range(iterations)),errors)
plt.show()

plt.scatter(data['population'],data['profit'])
h = theta[0] + theta[1]*x
plt.plot([min(x), max(x)], [min(h), max(h)], color='red')  # regression line
plt.show()




print ("Error =  ",errors[-1])

