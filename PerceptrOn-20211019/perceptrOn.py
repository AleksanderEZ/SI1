import numpy


class PerceptrOn(object):

    def __init__(self, p_eta=0.01, p_iterations_number=50):
        self.eta = p_eta
        self.iterations_number = p_iterations_number

    def fit(self, p_x, p_y):
        ramdon_seed = numpy.random.RandomState(seed=1)
        self.w_ = ramdon_seed.normal(loc=0, scale=0.01, size=1 + p_x.shape[1])
        self.updates_while_fit_ = []
        self.errors_while_fit_ = []
        for _ in range(self.iterations_number):
            updates = 0
            for xi, target in zip(p_x, p_y):
                individual_update = self.eta * (target - self.predict(xi))
                self.w_[1:] += individual_update * xi
                self.w_[0] += individual_update
                updates += int(individual_update != 0.0)
            self.updates_while_fit_.append(updates)
            self.errors_while_fit_.append(self.accuracy(p_x, p_y))
        return self

    def _net_input(self, p_x):
        return numpy.dot(p_x, self.w_[1:]) + self.w_[0]

    def predict(self, p_x):
        return numpy.where(self._net_input(p_x) >= 0.0, 1, -1)

    def accuracy(self, p_x, p_y):
        errors = 0
        for xi, target in zip(p_x, p_y):
            errors += int(target != self.predict(xi))
        return errors
