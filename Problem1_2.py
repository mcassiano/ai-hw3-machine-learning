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

    x_scaled = np.array(x)

    x_scaled[:,0] = ones
    x_scaled[:,1] = (x[:,1] - age_mean)/age_std
    x_scaled[:,2] = (x[:,2] - weight_mean)/weight_std
    
    m, n = x.shape

    # 2.3
    
    alphas = [0.005, 0.001, 0.05, 0.1, 0.5, 1.0]
    iterations_n = 50
    iterations = np.arange(iterations_n)
    risk = np.zeros(shape = (iterations_n, len(alphas))).T

    for alpha_i in range(0, len(alphas)):
        theta_sim = np.zeros(n)
        for iteration_n in iterations:
            theta_sim = lr.gradient_descent(x_scaled, y, theta_sim, alphas[alpha_i], iteration_n)
            prediction = np.dot(x_scaled, theta_sim)
            loss = prediction - y
            risk[alpha_i][iteration_n] = lr.calculate_cost(loss, m)

    for alpha_i in range(0, len(alphas)):
        plt.plot(iterations, risk[alpha_i], label='Alpha: %f' % alphas[alpha_i])

    theta = lr.gradient_descent(x_scaled,y,np.zeros(n), 1.0, 50)
    prediction = np.dot(x_scaled, theta)

    point_to_guess = [1.0, (5.0-age_mean)/age_std, (20.0-weight_mean)/weight_std]
    guess = np.sum(np.dot(theta, point_to_guess))

    print "The 5 year girl weighting 20 is approximately %f m tall." % (guess)

    plt.legend()
    plt.show()

    # 2.4
    
    p = np.matrix(x)
    theta = lr.normal_equation(p, y)

if __name__ == "__main__":
    main()