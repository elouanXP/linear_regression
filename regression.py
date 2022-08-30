import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

x,y = make_regression(n_samples=100,n_features=2,noise=10)

y = y.reshape(y.shape[0],1)
m = len(y)

X = np.hstack((x,np.ones((x.shape[0],1))))

theta = np.random.randn(3,1)

#modèle

def model (X,theta):
    return X.dot(theta)

#fonction coût

def cost_function (X,y,theta):
    return 1/(2*m) * np.sum((model(X,theta)-y)**2)

#gradient et descente de gradient
    
def grad (X,y,theta):
    return 1/m * X.T.dot(model(X,theta)-y)

def gradient_descent (X,y,theta,learning_rate,n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range (n_iterations):
        theta = theta - learning_rate * grad(X,y,theta)
        cost_history[i]=cost_function(X,y,theta)
    return theta,cost_history

theta_f,cost_history = gradient_descent(X,y,theta,learning_rate=0.01,n_iterations=800)

#analyse des résultats

predictions = model(X,theta_f)
plt.subplot(1,3,1)
plt.grid()
plt.scatter(x[:,0],y)
plt.scatter(x[:,0],predictions,c='r')

plt.subplot(1,3,2)
plt.grid()
plt.scatter(x[:,1],y)
plt.scatter(x[:,1],predictions,c='r')

plt.subplot(1,3,3)
plt.grid()
plt.plot(range(800),cost_history)

def coeff_determination (y,predic):
    a = ((y-predic)**2).sum()
    b = ((y-y.mean())**2).sum()
    return 1 - a/b

print(coeff_determination(y,predictions))

#3D

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x[:,0],x[:,1],y)
ax.scatter(x[:,0],x[:,1],predictions)
plt.tight_layout()



    
