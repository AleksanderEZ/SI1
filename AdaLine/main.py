import numpy as np
import Adaline
import matplotlib.pyplot as plt
import sklearn.datasets


def dataLoader():
    if iris:
        data = sklearn.datasets.load_iris(return_X_y=True)
        X = np.array(data[0][:, 2:])
        Y = np.array(data[1])
    else:
        X = np.array([[1.0, 2.0], [2.0, 1.0], [0, 3.0]])
        Y = np.array([-1, 1, -1])
    return X, Y


def plotErrorEvolution():
    a.fit(X, Y)
    ax1.plot(range(1, len(a.sse_) + 1), a.sse_, color='red', label='sse')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('SSE')
    ax1.set_title("SSE cost function")
    ax1.legend(loc='upper right')


def plotDataAndPrediction():
    classifier = a.w_
    print("weights", classifier)
    x = np.linspace(-plotBounds, plotBounds, 50)
    if iris:
        ax2.plot(x, -(classifier[0] + classifier[1] * x) / classifier[2], label='Prediction', color='red')
        ax2.scatter(setosa[:, 0], setosa[:, 1], label='Setosa', color='blue')
        ax2.scatter(versicolor[:, 0], versicolor[:, 1], label='Versicolor', color='orange')
        ax2.scatter(virginica[:, 0], virginica[:, 1], label='Virginica', color='green')
        ax2.set_xlabel('Petal length')
        ax2.set_ylabel('Petal width')
        ax2.legend(loc='upper right')
        ax2.axis([0, 8, 0, 3])
    else:
        ax2.plot(x, -(classifier[0] + classifier[1] * x) / classifier[2], label='Prediction', color='red')
        ax2.scatter(X[:, 0], X[:, 1], label='Samples')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend(loc='upper right')
        ax2.axis([-plotBounds, plotBounds, -plotBounds, plotBounds])
    ax2.set_title("Prediction")


def show():
    plt.show()


def lastOccurrenceOf(arr, element):
    return len(arr) - 1 - np.where(arr[::-1] == element)[0][0]


# Plot properties
plotBounds = 7
fig, (ax1, ax2) = plt.subplots(1, 2)

# DATA
iris = True
X, Y = dataLoader()
if iris:
    setosa = X[0:lastOccurrenceOf(Y, 0)+1, :]
    versicolor = X[lastOccurrenceOf(Y, 0)+2:lastOccurrenceOf(Y, 1)+1, :]
    virginica = X[lastOccurrenceOf(Y, 1)+2:lastOccurrenceOf(Y, 2)+1, :]
    X = X[0:lastOccurrenceOf(Y, 1)+1, :]
    Y = Y[0:lastOccurrenceOf(Y, 1)+1]

# TRAINING
delta = 0.0001
epochs = 500

a = Adaline.Adaline(p_eta=delta, p_iteration_number=epochs)
plotErrorEvolution()
plotDataAndPrediction()
show()
