import tensorflow as tf
from Graph import Graph

class GNG(object):
    def __init__(self, epsilon_b, epsilon_n, a_max, eta, alpha, delta):
        self.A = None
        self.N = []
        self.epsilon_b, self.epsilon_n, self.a_max, \
        self.eta, self.alpha, self.delta = epsilon_b, \
                                           epsilon_n, a_max, eta, alpha, delta

    def findFirstAndSecondNearestUnit(self, param):
        pass

    def fit(self, trainingX, epochs):
        # Paso 1
        self.A = tf.Variable(tf.random.normal([2, trainingX.shape[1]], 0.0, 1.0, dtype=tf.float32))
        self.N.append(Graph(0))
        self.N.append(Graph(1))
        self.error_ = tf.Variable(tf.zeros([trainingX.shape[1]]))

        for epoch in tf.range(epochs):
            shuffledTrainingX = tf.random.shuffle(trainingX)
            for row_ in tf.range(shuffledTrainingX.shape(0)):
                # Paso 2
                xi = shuffledTrainingX[row_]
                # Paso 3
                indexS1, indexS2 = self.findFirstAndSecondNearestUnit(xi)
                # Paso 4
                self.N[indexS1].increaseAgeOfNeighbours()
                # Paso 5
                self.error_.assign(self.error_ + tf.math.reduce_sum(tf.math.squared_difference(xi, shuffledTrainingX[indexS1])))
                # Paso 6
                self.A[indexS1].assign(self.A[indexS1] + self.epsilon_b * (xi - shuffledTrainingX[indexS1]))
                self.A[self.N[indexS1].neighborhood].assign(self.A[self.N[indexS1].neighborhood] + self.epsilon_n * (xi - shuffledTrainingX[self.N[indexS1].neighborhood]))
                # Paso 7
