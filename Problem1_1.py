import matplotlib.pyplot as plt
import numpy as np
import LinearRegression as lr

def main():

	# 1.1
	data = np.genfromtxt('girls_train.csv', delimiter=',')

	ones = np.ones(len(data))
	x = np.array([ones, data[:,0]]).transpose()
	y = data[:,1]

	plt.scatter(x[:,1], y)

	# 1.2
	
	m, n = x.shape
	learning_rate = 0.05
	number_of_iterations = 1500
	theta = np.ones(n)
	theta, history = lr.gradient_descent(x, y, theta, learning_rate, m, number_of_iterations)

	prediction = np.dot(x, theta)
	
	plt.plot(x[:,1], prediction)
	plt.show()


if __name__ == "__main__":
	main()