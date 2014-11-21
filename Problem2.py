import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def plot_decision_boundary(model, subplot, classes, labels, x_train, y_train):
    '''
        Plots the decision boundary using the values
        of training data and scatter the whole data set
    '''

    h = 0.02

    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    subplot.contourf(xx, yy, Z, cmap=plt.cm.Paired, zorder=1)
    subplot.scatter(classes[:, 0], classes[:, 1], c=labels, cmap=plt.cm.Paired)


def main():

    data = np.genfromtxt('chessboard.csv', delimiter=',')[1:]
    classes = data[:,:2]
    labels = data[:,2]

    # splits the data into 60% training and 40% testing
        
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        classes, labels, test_size=0.4, random_state=0)

    # settings variables to train data
    
    k_folds = [5, 10]
    k_neighbors = [3,4,5,6] #[1,2,3,4,5,6,7,8]
    degrees = [1,2,3]#[1,2,3,4,5,6]
    gammas = [10]#[0.01, 0.1, 1, 10]
    cs = [1]#0.01, 0.1, 1, 10]

    params_svm_knn = {
        'rbf': [{'kernel': ['rbf'], 'gamma': gammas, 'C': cs}],
        'linear': [{'kernel': ['linear'], 'C': cs}],
        'poly': [{'kernel': ['poly'], 'C': cs, 'degree': degrees}],
        'knn': [{'n_neighbors': k_neighbors}]
    }

    other_classifiers = [
        (RandomForestClassifier(), 'Random Forest'),
        (AdaBoostClassifier(), 'ADA Boost'),
        (DecisionTreeClassifier(), 'Decision Tree')
    ]

    score_history = []

    scores = {}
    for classifier, params in params_svm_knn.items():
        scores[classifier] = []

    for k in k_folds:
        for classifier, params in params_svm_knn.items():

            if classifier != 'knn':
                clf = GridSearchCV(svm.SVC(), params, cv=k)
                clf.fit(X_train, y_train)
                scores[classifier].append([clf.best_score_, k, clf.best_params_])
                score_history.append(clf.grid_scores_)

            else:
                clf = GridSearchCV(KNeighborsClassifier(), params, cv=k)
                clf.fit(X_train, y_train)
                scores[classifier].append([clf.best_score_, k, clf.best_params_])
                score_history.append(clf.grid_scores_)
        

    best_rbf = max(scores['rbf'], key = lambda s: s[0])
    best_linear = max(scores['linear'], key = lambda s: s[0])
    best_poly = max(scores['poly'], key = lambda s: s[0])
    best_knn = max(scores['knn'], key = lambda s: s[0])

    main_graphs = plt.figure()
    distribution = main_graphs.add_subplot(2, 2, 1)
    graph_linear = main_graphs.add_subplot(2, 2, 2)
    graph_poly = main_graphs.add_subplot(2, 2, 3)
    graph_rbf = main_graphs.add_subplot(2, 2, 4)


    # pass the dictionary as kwargs
    lin = svm.SVC(**best_linear[2]).fit(X_train, y_train)
    poly = svm.SVC(**best_poly[2]).fit(X_train, y_train)
    rbf = svm.SVC(**best_rbf[2]).fit(X_train, y_train)
    knn = KNeighborsClassifier(**best_knn[2]).fit(X_train, y_train)

    graph_extra_params = {
        'fontsize': 10
    }

    distribution.scatter(classes[:,0], classes[:,1],
        c=labels, marker='o', zorder=2)
    distribution.set_title('Data distribution', **graph_extra_params)

    plot_decision_boundary(lin, graph_linear, classes, labels, 
        X_train, y_train)

    graph_linear.set_title('Linear. c = %0.2f' % 
        (best_linear[2]['C']), **graph_extra_params)

    plot_decision_boundary(poly, graph_poly, classes, labels, 
        X_train, y_train)

    graph_poly.set_title('Polynomial. c = %0.2f degree = %0.2f' %
        (best_poly[2]['C'], best_poly[2]['degree']), **graph_extra_params)

    plot_decision_boundary(rbf, graph_rbf, classes, labels, 
        X_train, y_train)

    graph_rbf.set_title('RBF. c = %0.2f gamma = %0.2f' %
        (best_rbf[2]['C'], best_rbf[2]['gamma']), **graph_extra_params)


    extra_graphs = plt.figure()
    graph_knn = extra_graphs.add_subplot(2, 2, 1)

    plot_decision_boundary(knn, graph_knn, classes, labels, 
        X_train, y_train)

    graph_knn.set_title('KNN. k = %d' %
        (best_knn[2]['n_neighbors']), **graph_extra_params)    

    for i in range(2, 5):
        classifier = other_classifiers[i-2][0]
        classifier.fit(X_train, y_train)

        classifier_graph = extra_graphs.add_subplot(2,2,i)
        classifier_graph.set_title(other_classifiers[i-2][1],
            **graph_extra_params)

        plot_decision_boundary(classifier, classifier_graph, classes, labels,
                X_train, y_train)

        score_history.append(classifier.score(X_train, y_train))


    plt.show()

if __name__ == "__main__":
    main()