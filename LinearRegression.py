import numpy as np

class LinearRegression:

    def calculate_cost(self, loss, m):
        '''
            Calculates the risk (cost?) of a given
            vector loss = (prediction - y)
        '''
        return np.sum(loss ** 2.0) / (2.0 * m)

    def gradient_descent(self, x, y, theta, alpha, numIterations):
        '''
            Performs gradient descent (multivariate) on x and y
            variating theta with a learning rate alpha. Runs
            numIterations times.
        '''

        m = x.shape[0]
        xTrans = x.transpose()

        for i in range(0, numIterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = np.dot(xTrans, loss) / m
            theta = theta - alpha * gradient

        return theta

    def normal_equation(self, x, y):
        '''
            Performs the Normal Equation approach.
            theta = (x^T . x)^-1 . x^T . y 
        '''
        return np.dot(np.dot((x.T.dot(x)).I, x.T), y)