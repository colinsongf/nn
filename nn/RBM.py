import numpy as np
import time
from Functions import sigmoid
from Distances import euclidian
from Helpers import neuron_local_gain_update


class RBM:
    W = None # weights of network, rows are weights of visible layer
    a = None # biases of visible layer
    b = None # biases of hidden layer
    use_biases = True

    def __str__(self):
        return "RBM: " + str(self.W.shape[0]) + '<->' + str(self.W.shape[1])


    def __init__(self,
                 visible_size,
                 hidden_size,
                 rng=(lambda n: np.random.normal(0, 0.1, n)),
                 use_biases=True
    ):
        self.W = rng(visible_size * hidden_size).reshape((visible_size, hidden_size))
        self.a = rng(visible_size)
        self.b = rng(hidden_size)
        self.use_biases = use_biases

    def sample(self, m):
        return (np.random.uniform(0, 1, m.shape) < m).astype(float)

    def compute_output(self, input_data, do_sampling=True):
        if self.use_biases:
            h = sigmoid(np.dot(np.c_[np.ones(input_data.shape[0]), input_data], np.vstack((self.b, self.W))))
        else:
            h = sigmoid(np.dot(input_data, self.W))
        if do_sampling:
            h = self.sample(h)
        return h


    def generate_input(self, input_data, do_sampling=False):
        if self.use_biases:
            v = sigmoid(np.dot(np.c_[np.ones(input_data.shape[0]), input_data], np.vstack((self.a, self.W.T))))
        else:
            v = sigmoid(np.dot(input_data, self.W.T))
        if do_sampling:
            v = self.sample(v)
        return v

    def train(self,
              input_data,
              cd_k = 1,
              learning_rate = 0.1,
              momentum_rate = 0.9,
              max_iter = 10000,
              batch_size=20,
              stop_threshold=0.15,
              goal = euclidian,
              cv_input_data = None,
              regularization_rate = 0.1,
              regularization_norm = None,
              d_regularization_norm = None,
              neural_local_gain = None,
              do_visible_sampling=False,
              n_iter_stop_skip=10,
              min_train_cost = np.finfo(float).eps,
              min_cv_cost = np.finfo(float).eps,
              tolerance=0,
              verbose=True):
        nlg_W = None
        nlg_a = None
        nlg_b = None
        if neural_local_gain is not None:
            nlg_bonus, nlg_penalty, nlg_min, nlg_max = neural_local_gain
            nlg_W = np.ones(self.W.shape)
            nlg_a = np.ones(self.a.shape)
            nlg_b = np.ones(self.b.shape)
        cost = []
        cv_cost = []
        last_delta_W = np.zeros(self.W.shape)
        last_delta_a = np.zeros(self.a.shape)
        last_delta_b = np.zeros(self.b.shape)
        for n_iter in range(max_iter):
            t_start = time.clock()
            idx_data = range(input_data.shape[0])
            np.random.shuffle(idx_data)
            idx_batch = range(0, len(idx_data), batch_size)
            for i_in_batch in idx_batch:
                batch_data = input_data[idx_data[i_in_batch:(i_in_batch + batch_size)], :]
                v = batch_data
                nabla_W = np.zeros(self.W.shape)
                nabla_a = np.zeros(self.a.shape)
                nabla_b = np.zeros(self.b.shape)
                for k in range(cd_k + 1):
                    h = self.compute_output(v, do_sampling=False)
                    if k == cd_k:
                        # accumulate negative phase
                        nabla_W -= np.dot(v.T, h)
                        if self.use_biases:
                            nabla_a -= np.sum(v, axis=0)
                            nabla_b -= np.sum(h, axis=0)
                        break
                    h = self.sample(h)
                    if k == 0:
                        # accumulate positive phase
                        nabla_W += np.dot(v.T, h)
                        if self.use_biases:
                            nabla_a += np.sum(v, axis=0)
                            nabla_b += np.sum(h, axis=0)

                    v = self.generate_input(h, do_sampling=False)
                    if do_visible_sampling:
                        v = self.sample(v)
                nabla_W /= batch_size
                nabla_a /= batch_size
                nabla_b /= batch_size

                # update weights
                regularization_penalty = 0.0 if d_regularization_norm is None else d_regularization_norm(self.W)
                delta_W = (learning_rate if nlg_W is None else learning_rate * nlg_W) * (
                    momentum_rate * last_delta_W +
                    nabla_W -
                    (0.0 if d_regularization_norm is None else regularization_rate * regularization_penalty)
                )
                if nlg_W is not None:
                        c = delta_W * last_delta_W >= 0
                        nlg_W = neuron_local_gain_update(nlg_W, c, nlg_bonus, nlg_penalty, nlg_min, nlg_max)
                self.W += delta_W
                last_delta_W = delta_W
                if self.use_biases:
                    delta_a = (learning_rate if nlg_a is None else learning_rate * nlg_a) * (
                        momentum_rate * last_delta_a +
                        nabla_a
                    )
                    if nlg_a is not None:
                        c = delta_a * last_delta_a >= 0
                        nlg_a = neuron_local_gain_update(nlg_a, c, nlg_bonus, nlg_penalty, nlg_min, nlg_max)
                    self.a += delta_a
                    last_delta_a = delta_a
                    delta_b = (learning_rate if nlg_b is None else learning_rate * nlg_b) * (
                        momentum_rate * last_delta_b +
                        nabla_b
                    )
                    if nlg_b is not None:
                        c = delta_b * last_delta_b >= 0
                        nlg_b = neuron_local_gain_update(nlg_b, c, nlg_bonus, nlg_penalty, nlg_min, nlg_max)
                    self.b += delta_b
                    last_delta_b = delta_b

            # compute cost
            cost.append(np.sum(goal(input_data,
                                    self.generate_input(self.compute_output(input_data, do_sampling=True), do_sampling=True))
            ) / input_data.shape[0])
            if cv_input_data is not None:
                cv_cost.append(np.sum(goal(cv_input_data,
                                    self.generate_input(self.compute_output(cv_input_data, do_sampling=True), do_sampling=True))
                ) / cv_input_data.shape[0])

            t_total = time.clock() - t_start

            if verbose:
                if len(cv_cost) > 0:
                    print('Iteration: %s (%s s), train/cv cost: %s / %s' % (n_iter, t_total, cost[-1], cv_cost[-1]))
                else:
                    print('Iteration: %s (%s s), train cost = %s' % (n_iter, t_total, cost[-1]))

            if n_iter > n_iter_stop_skip:
                if len(cv_cost) > 0:
                    if (1 - stop_threshold) * cv_cost[-1] > min(cv_cost):
                        break
                else:
                    if (1 - stop_threshold) * cost[-1] > min(cost):
                        break
                if cost[-1] <= min_train_cost:
                    break
                if len(cv_cost) > 0 and cv_cost[-1] <= min_cv_cost:
                    break
                if len(cost) > 1 and abs(cost[-1] - cost[-2]) < tolerance:
                    break



