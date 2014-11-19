import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from LinearRegression import LinearRegression

def main():

    # 2
    data = np.genfromtxt('girls_age_weight_height_2_8.csv', delimiter=',')
    lr = LinearRegression()

    ones = np.ones(len(data))
    x = np.array([ones, data[:,0], data[:,1]]).transpose()
    y = data[:,2]

    # 2.1

    age_mean = np.mean(x[:,1])
    age_std = np.std(x[:,1])

    weight_mean = np.mean(x[:,2])
    weight_std = np.std(x[:,2])

    print "Feature 'age' - Mean: %f, STD: %f" % (age_mean, age_std)
    print "Feature 'weight' - Mean: %f, STD: %f" % (weight_mean, weight_std)

    x[:,1] = (x[:,1] - age_mean)/age_std
    x[:,2] = (x[:,2] - weight_mean)/weight_std

    # 2.2
    
    m, n = x.shape
    learning_rate = 0.05
    number_of_iterations = 1500
    theta = np.zeros(n)
    theta = lr.gradient_descent(x, y, theta, learning_rate, number_of_iterations)

    # 2.3
    
    alphas = [0.005, 0.001, 0.05, 0.1, 0.5, 1]





if __name__ == "__main__":
    main()