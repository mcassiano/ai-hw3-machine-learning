import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def plot_data(model, fig, x, y):

    h = 0.02

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig.contourf(xx, yy, Z, cmap=plt.cm.Paired, zorder=1)
    fig.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)

    #fig.xlim(xx.min(), xx.max())
    #fig.ylim(yy.min(), yy.max())

def main():

    data = np.genfromtxt('chessboard.csv', delimiter=',')[1:]
    classes = data[:,:2]
    labels = data[:,2]

    fig = plt.figure()

    data_distribution = fig.add_subplot(2, 3, 1)
    data_distribution.scatter(classes[:,0], classes[:,1], c=labels, marker='o', zorder=2)
    data_distribution.set_title('Data distribuition', fontsize=12)
    
    graph_linear = fig.add_subplot(2, 3, 2)
    graph_poly = fig.add_subplot(2, 3, 3)
    graph_rbf = fig.add_subplot(2, 3, 4)
    graph_knn = fig.add_subplot(2,3, 5)


    # setting variables for data splitting
    
    h = .02
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        classes, labels, test_size=0.4, random_state=0)

    # settings variables for training data
    
    k_folds = [5, 10]
    k_neighbors = [1,2,3,4,5,6,7,8]
    degrees = [5,6]
    gammas = [0.01, 0.1, 1, 10]
    cs = [0.1, 1]

    lin_scores = []
    poly_scores = []
    rbf_scores = []
    knn_scores = []        

    for k in k_folds:

        for kn in k_neighbors:
            knn = KNeighborsClassifier(n_neighbors=kn)
            scores = cross_validation.cross_val_score(
                        knn, X_train, y_train, cv=k)
            knn_scores.append([scores.mean(), k, kn])

        for c in cs:
            linear_svc = svm.SVC(kernel='linear', C=c)
            scores = cross_validation.cross_val_score(
                linear_svc, X_train, y_train, cv=k)
            lin_scores.append([scores.mean(), k, c])

            for gamma in gammas:
                rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=c)
                scores = cross_validation.cross_val_score(
                    rbf_svc, X_train, y_train, cv=k)
                rbf_scores.append([scores.mean(), k, c, gamma])

            for degree in degrees:
                poly_svc = svm.SVC(kernel='poly', degree=degree, C=c)
                scores = cross_validation.cross_val_score(
                    poly_svc, X_train, y_train, cv=k)
                poly_scores.append([scores.mean(), k, c, degree])

    
    best_linear = max(lin_scores, key = lambda s: s[0])
    best_poly = max(poly_scores, key = lambda s: s[0])
    best_rbf = max(rbf_scores, key = lambda s: s[0])
    best_knn = max(knn_scores, key = lambda s: s[0])

    lin = svm.SVC(kernel='linear', C=best_linear[2]).fit(X_test, y_test)
    poly = svm.SVC(kernel='poly', degree=best_poly[3], C=best_poly[2]).fit(X_test, y_test)
    rbf = svm.SVC(kernel='rbf', gamma=best_rbf[3], C=best_rbf[2]).fit(X_test, y_test)
    knn = KNeighborsClassifier(n_neighbors=best_knn[2]).fit(X_train, y_train)

    plot_data(lin, graph_linear, X_test, y_test)
    graph_linear.set_title("Linear. C = %0.1f" % best_linear[2], fontsize=12)

    plot_data(poly, graph_poly, X_test, y_test)
    graph_poly.set_title("Poly. C = %0.1f, d = %d" % (best_poly[2], best_poly[3]), fontsize=12)

    plot_data(rbf, graph_rbf, X_test, y_test)
    graph_rbf.set_title("RBF C = %0.1f, g = %0.1f" % (best_rbf[2], best_rbf[3]), fontsize=12)

    plot_data(knn, graph_knn, X_test, y_test)
    graph_knn.set_title("K-Nearest neighbors. K = %d" % best_knn[2], fontsize=12)


    plt.show()

if __name__ == "__main__":
    main()