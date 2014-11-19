import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from LinearRegression import LinearRegression

def main():

    # 1.1
    data = np.genfromtxt('girls_train.csv', delimiter=',')
    lr = LinearRegression()

    ones = np.ones(len(data))
    x = np.array([ones, data[:,0]]).transpose()
    y = data[:,1]

    #plt.scatter(x[:,1], y)

    # 1.2
    
    m, n = x.shape
    learning_rate = 0.05
    number_of_iterations = 1500
    theta = np.zeros(n)
    theta, history, cost = lr.gradient_descent(x, y, theta, learning_rate, number_of_iterations)

    X, Y = np.meshgrid(history[:,0], history[:,1])
    Z = np.ones(shape=X.shape)*cost.T

    prediction = np.dot(x, theta)
    
    # 1.3
    
    #plt.plot(x[:,1], prediction)
    #plt.show()
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(X, Y, Z)

    # 1.4
    
    print "Model: %fx + %f" % (theta[1], theta[0])
    print "Mean square error for training: %f" % cost[-1]
    print "Predicted height for a 4.5 years old girl: %f" % (4.5*theta[1] + theta[0])

    test_data = np.genfromtxt('girls_test.csv', delimiter=',')
    test_x = np.array([np.ones(len(test_data)), test_data[:,0]]).transpose()
    test_y = test_data[:,1]
    test_prediction = np.dot(test_x, theta)
    error = lr.calculate_cost(test_prediction - test_y, len(data))
    
    print "Mean square error for testing: %f" % (error)


    #plt.show()


if __name__ == "__main__":
    main()