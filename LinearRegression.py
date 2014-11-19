import numpy as np

class LinearRegression():

    def calculate_cost(x, y, theta, m):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        return np.sum(loss ** 2) / (2 * m)

    def gradient_descent(x, y, theta, alpha, m, numIterations):
        xTrans = x.transpose()
        theta_history = np.zeros(shape = (numIterations, x.shape[1]))
        cost_history = np.zeros(shape = (numIterations, 1))

        for i in range(0, numIterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = np.dot(xTrans, loss) / m
            theta = theta - alpha * gradient
            theta_history[i] = theta
            #cost_history[i] = calculate_cost(loss, m)

        return theta, theta_history#, cost_history