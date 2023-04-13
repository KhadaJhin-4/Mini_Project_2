# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn import tree, datasets
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier


def DTPrint(result):
    if result == 1:
        iris = load_iris()
        X, y = iris.data, iris.target
    elif result == 2:
        digits = load_digits()
        X, y = digits.data, digits.target
    else:
        pass

    dtc = tree.DecisionTreeClassifier(criterion="entropy")
    dtc.fit(X, y)
    tree.plot_tree(dtc)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_pred = dtc.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))


def PerceptonPrint(result, kfold):
    if result == 1:
        iris = datasets.load_iris()
        n_samples = len(iris.data)
        if kfold == 0:
            data = iris.data
            percepton(data, iris, result)
        elif kfold == 1:
            X = iris.data
            Y = iris.target
            Kfold(X, Y)
        else:
            pass
    elif result == 2:
        digits = datasets.load_digits()
        n_samples = len(digits.images)
        if kfold == 0:
            data = digits.images.reshape((n_samples, -1))
            percepton(data, digits, result) #if you pay attention, data and digits is the same value as X and Y, but they're used differently inside their called functions
        elif kfold == 1:
            X = digits.images.reshape((n_samples, -1))
            Y = digits.target
            Kfold(X, Y)
        else:
            pass
    else:
        pass


def percepton(data, dataset, result):
    X_train, X_test, y_train, y_test = train_test_split(data, dataset.target, test_size=0.3,
                                                        shuffle=False)  # test with 30%, the rest is training

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4, solver='sgd', verbose=10,
                        random_state=1, learning_rate_init=0.001)  # training a neural network to make predictions
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    pre_macro = precision_score(y_test, y_pred, average='macro')
    pre_micro = precision_score(y_test, y_pred, average='micro')

    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_micro = recall_score(y_test, y_pred, average='micro')

    cm = confusion_matrix(y_test, y_pred)
    if result == 1:
        print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    else:
        ConfusionMatrixDisplay(cm, display_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).plot()
        plt.show()



def Kfold(X, Y):
    kf = KFold(n_splits=10)
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4, solver='sgd', verbose=10,
                        random_state=1, learning_rate_init=0.001)  # training a neural network to make predictions
    for train_index, test_index in kf.split(X, Y):
        x_train_fold = X[train_index]
        y_train_fold = Y[train_index]
        x_test_fold = X[test_index]
        y_test_fold = Y[test_index]
        mlp.fit(x_train_fold, y_train_fold)
        print(mlp.score(x_test_fold, y_test_fold))


def printDiff():
    iris = load_iris()
    X, y = iris.data, iris.target
    dtc = tree.DecisionTreeClassifier()

    dot_data = tree.export_graphviz(dtc, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("iristree")


if __name__ == '__main__':
    result = 2 #user defined, 1 is for iris, 2 is for digits
    kfold = 1 # user defined, if kfold = 1, it runs with kfold version inside of PerceptonPrint()

    # DTPrint(result)
    # #printDiff()

    PerceptonPrint(result, kfold)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
