import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn import svm

def main():

    data = np.genfromtxt('chessboard.csv', delimiter=',')[1:]
    classes = data[:,:2]
    labels = data[:,2]

    #plt.subplot(2, 2, 1)
    #plt.scatter(classes[:,0], classes[:,1], c=labels, marker='o', s=40, zorder=2)


    # setting variables for data splitting
    
    h = .02
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        classes, labels, test_size=0.4, random_state=0)

    #lin_svc = svm.LinearSVC(C=0.05).fit(X_train, y_train)
    #poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
    
    # RBF kernel

    c_values = np.arange(1,1000, step=100)
    gamma_values = np.arange(0.01, 10.0, 0.5)
    rbf_scores = []

    for c in c_values:
        for gamma in gamma_values:
            rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=c).fit(X_train, y_train)
            score = rbf_svc.score(X_test, y_test)
            rbf_scores.append([score, c, gamma])

    score, c, gamma = max(rbf_scores, key=lambda s:s[0])
    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=c).fit(X_train, y_train)


    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    
    Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, zorder=1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

    #plt.show()

if __name__ == "__main__":
    main()