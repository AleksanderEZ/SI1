import tensorflow as tf

class GNG(object):
    def __init__(self, ):
        self.A = None
        self.N = []

    def fit(self, X, epochs):
        self.ab = tf.random.normal([2, X.shape[1]], 0.0, 1.0, dtype=tf.float32)

        for epoch in tf.range(epochs):
            pass