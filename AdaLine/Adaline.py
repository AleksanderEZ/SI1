import numpy


class Adaline(object):

    def __init__(self, p_eta=1, p_iteration_number=50, p_random_state=1):
        self.eta = p_eta
        self.iterations_number = p_iteration_number
        self.random_state = p_random_state

    def fit(self, p_x, p_y):
        random_seed = numpy.random.RandomState(seed=self.random_state)
        self.w_ = random_seed.normal(loc=0.0, scale=0.01, size=1 + p_x.shape[1])
        self.sse_ = []

        for _ in range(self.iterations_number):
            v_errors = p_y - self._activation(self._net_input(p_x))
            self.w_[1:] += self.eta * p_x.T.dot(v_errors)
            self.w_[0] += self.eta * v_errors.sum()
            self.sse_.append(0.5 * (v_errors ** 2).sum())

        return self

    def _net_input(self, p_x):
        return numpy.dot(p_x, self.w_[1:] + self.w_[0])

    def _activation(self, p_net_input):
        return p_net_input

    def _quantization(self, p_activation):
        return numpy.where(p_activation >= 0.0, 1, -1)

    def predict(self, p_x):
        return self._quantization(self._activation(self._net_input(p_x)))
