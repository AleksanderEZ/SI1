import numpy
import pandas
import matplotlib.pyplot
import perceptrOn

def test_0():
    p = perceptrOn.PerceptrOn(p_eta=0.01, p_iterations_number=10)
    v_x = numpy.array([[1.0, 2.0], [2.0, 1.0], [1.5, 5]])
    v_y = numpy.array([-1, 1, -1])
    p.fit(v_x, v_y)
    print(p.updates_while_fit_)

def test_1():
    df = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # print(df.head(3))
    # print(df.tail(3))

    y_data = df.iloc[0:, 4].values
    y = numpy.where(y_data == 'Iris-setosa', -1, numpy.where(y_data == 'Iris-virginica', 1, 0))

    x = df.iloc[0:, [0, 2]].values

    x_data_0 = x[y < 0, :]
    x_data_1 = x[y > 0, :]
    x_data_2 = x[y == 0, :]

    X = numpy.concatenate((x_data_1[:, :], x_data_2[:, :]))
    Y = numpy.concatenate((numpy.ones(x_data_1.shape[0]) * -1, numpy.ones(x_data_2.shape[0])))

    p = perceptrOn.PerceptrOn(p_eta=0.01, p_iterations_number=20)
    p.fit(X, Y)

    print(p.updates_while_fit_)
    print(p.errors_while_fit_)

    punto_0_x = 4
    punto_0_y = - (p.w_[1] * punto_0_x + p.w_[0]) / p.w_[2]
    punto_1_x = 9
    punto_1_y = - (p.w_[1] * punto_1_x + p.w_[0]) / p.w_[2]

    matplotlib.pyplot.plot([punto_0_x, punto_1_x], [punto_0_y, punto_1_y])

    matplotlib.pyplot.scatter(x_data_0[:, 0], x_data_0[:, 1], color='red', marker='o', label='setosa')
    matplotlib.pyplot.scatter(x_data_1[:, 0], x_data_1[:, 1], color='blue', marker='x', label='virginica')
    matplotlib.pyplot.scatter(x_data_2[:, 0], x_data_2[:, 1], color='black', marker='*', label='versicolor')

    matplotlib.pyplot.xlabel('sepal length [cm]')
    matplotlib.pyplot.ylabel('petal length [cm]')
    matplotlib.pyplot.legend(loc='upper left')
    matplotlib.pyplot.show()

test_1()