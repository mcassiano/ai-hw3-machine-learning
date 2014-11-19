import numpy as np

class LinearRegression:

    def calculate_cost(self, loss, m):
        return np.sum(loss ** 2.0) / (2.0 * m)

    def gradient_descent(self, x, y, theta, alpha, numIterations):
        m = x.shape[0]
        xTrans = x.transpose()
        #theta_history = np.zeros(shape = (numIterations, x.shape[1]))
        #cost_history = np.zeros(shape = (numIterations, 1))

        for i in range(0, numIterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = np.dot(xTrans, loss) / m
            theta = theta - alpha * gradient
            #theta_history[i] = theta
            #cost_history[i] = self.calculate_cost(loss, m)

        return theta#, theta_history, cost_history