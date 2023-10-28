# Gradient descent is used to minimize a cost function, which measures the difference between the predicted and actual values in a machine learning model.
# cost function = actuall price - predicted 
# The purpose of this code is to iteratively update the parameters m and b to find the values that minimize the Mean Squared Error, thus providing a linear model that best fits the given data points (x and y). 
# mean square error func: MSE = (1 / n) * Σ(yi - (mxi + b))^2
# partial derivate to m = (2 / n) * Σ xi(yi - (mxi + b))
# partial derivate to b = (2 / n) * Σ xi(yi - (mxi + b))

# partial derviated f= x^2+ y^3 , d/dx = 2x and d/dy = 3y^2
# learning rate = m - learning_rate * d/dm ad so is the b 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(x,y):
    m_curr = b_curr = 0
    # change the itr value to see the curvature of the graph
    itr = 3000
    n = len(x)
    learning_rate = 0.001
    costs = []
    iterations = []


    for i in range(itr):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        m_derivate = -(2/n) * sum(x*(y - y_predicted))
        b_derivate = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * m_derivate
        b_curr = b_curr - learning_rate * b_derivate
        print ("m{},b{},cost{},itreation{}".format(m_curr,b_curr,cost,i))
        
        # Append cost and iteration to the lists
        costs.append(cost)
        iterations.append(i)

    # Plot the cost vs. iteration
    plt.plot(iterations, costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iteration')
    plt.show()

df = pd.read_csv("homeprices.csv")
x = df['age']
y = df['price']

gradient_descent(x,y)