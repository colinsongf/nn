import numpy as np
import time
from Functions import sigmoid
from Distances import *
from Helpers import neuron_local_gain_update


class RBM:
    """
    Restricted Boltzmann Machine (RBM) with contrastive divergence training algorithm
    """

    W = None # weights of network, rows are weights of visible layer
    a = None # biases of visible layer
    b = None # biases of hidden layer
    mode = 'bin-bin'
    use_biases = True

    def __str__(self):
        return "RBM: " + str(self.W.shape[0]) + ' <-> ' + str(self.W.shape[1])


    def __init__(self,
                 visible_size,
                 hidden_size,
                 rng=(lambda n: np.random.normal(0, 0.1, n)),
                 mode='bin-bin', # gaus-bin (gaussian <-> bernoulli), bin-bin (bernoulli <-> bernoulli or binary-binary)
                 use_biases=True
    ):
        """
        Initialization of RBM
        :param visible_size: input space dimension
        :param hidden_size: hidden space dimension
        :param rng: function that returns numpy array of n random numbers
        :param mode: mode of network: gaus-bin (gaussian <-> bernoulli), bin-bin (bernoulli <-> bernoulli or binary-binary)
        :param use_biases: use biases in calculations
        :return:
        """
        self.W = rng(visible_size * hidden_size).reshape((visible_size, hidden_size))
        self.a = rng(visible_size)
        self.b = rng(hidden_size)
        self.use_biases = use_biases
        if mode not in ['bin-bin', 'gaus-bin']:
            raise Exception('unsupported mode')
        else:
            self.mode = mode

    def sample(self, m):
        """
        Sample 0 or 1 with respect to given matrix of probabilities
        :param m: matrix of probabilities
        :return: binary matrix with same shape
        """
        return (np.random.uniform(0, 1, m.shape) < m).astype(float)

    def compute_output(self, input_data, do_sampling=True):
        """
        Compute hidden state
        :param input_data: numpy ndarray
        :param do_sampling: do binary sample or not
        :return: data representation in hidden space
        """
        if len(input_data.shape) == 1:
            input_data.shape = (1, input_data.shape[0])
        if self.use_biases:
            h = sigmoid(np.dot(np.c_[np.ones(input_data.shape[0]), input_data], np.vstack((self.b, self.W))))
        else:
            h = sigmoid(np.dot(input_data, self.W))
        if do_sampling:
            h = self.sample(h)
        return h


    def generate_input(self, input_data, do_sampling=False):
        """
        Restore input data using hidden space representation
        :param input_data: data representation in hidden space
        :param do_sampling: do_sampling: do binary sample or not (doesn't matter in gaus-bin mode)
        :return: data representation in original space
        """
        if self.mode == 'gaus-bin':
            return np.dot(np.c_[np.ones(input_data.shape[0]), input_data], np.vstack((self.a, self.W.T)))
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
              is_sparse=False,
              verbose=True):
        """
        Train RBM with contrastive divergence algorithm
        :param input_data: numpy ndarray
        :param cd_k: number of iterations in Gibbs sampling
        :param learning_rate: small number
        :param momentum_rate: small number
        :param max_iter: maximum number of iteration
        :param batch_size: 1 - online learning, n < input_data.shape[0] - batch learning,
        None or input_data.shape[0] - full batch learning
        :param stop_threshold: stop training if (1 - stop_threshold)*(current cost) > min(cost);
        is crossvalidation set presented then cv_cost is used
        :param goal: train cost function f: ndarray_NxM -> array_N, N - number of examples;
         in other words it returns cost for each of examples
        :param cv_input_data: numpy ndarry
        :param regularization_rate: small number
        :param regularization_norm: norm to penalize model, if set then it will be included in cost
        :param d_regularization_norm: derivative of norm to penalize model, if set then it will be takken into account
        in error calculation
        :param neural_local_gain: tuple (bonus, penalty, min, max), type of adaptive learning rate;
        https://www.cs.toronto.edu/~hinton/csc321/notes/lec9.pdf
        :param do_visible_sampling: do sampling of visible units in contrastive divergence,
        http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf 3.2 Updating the visible states (page 6)
        :param n_iter_stop_skip: number of iteration skip shecking of stop conditions
        :param min_train_cost: minimum value of cost on train set
        :param min_cv_cost: minimum value of cost on crossvalidation set
        :param tolerance: minimum step of cost
        :param verbose: logging
        :return: values of cost in each of iterations, if cv_cost is set then also returned list of cv_goal values
        """
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
                batch_data = input_data[idx_data[i_in_batch:(i_in_batch + batch_size)], :] if not is_sparse \
                    else np.asarray(input_data[idx_data[i_in_batch:(i_in_batch + batch_size)], :].todense())
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
                            if self.mode == 'bin-bin':
                                nabla_a -= np.sum(v, axis=0)
                            elif self.mode == 'gaus-bin':
                                nabla_a -= np.sum(np.repeat(self.a, v.shape[0]).reshape((v.shape[0], v.shape[1]), order='F'), axis=0)
                            nabla_b -= np.sum(h, axis=0)
                        break
                    h = self.sample(h)
                    if k == 0:
                        # accumulate positive phase
                        nabla_W += np.dot(v.T, h)
                        if self.use_biases:
                            if self.mode == 'bin-bin':
                                nabla_a += np.sum(v, axis=0)
                            elif self.mode == 'gaus-bin':
                                nabla_a += np.sum(np.repeat(self.a, v.shape[0]).reshape((v.shape[0], v.shape[1]), order='F'), axis=0)
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
            if not is_sparse:
                cost.append(np.sum(goal(input_data,
                                        self.generate_input(self.compute_output(input_data, do_sampling=True), do_sampling=True)) +
                                   (0.0 if regularization_norm is None else regularization_norm(self.W))
                ) / input_data.shape[0])
            if cv_input_data is not None:
                cv_cost.append(np.sum(goal(np.asarray(cv_input_data.todense()),
                                    self.generate_input(self.compute_output(np.asarray(cv_input_data.todense()), do_sampling=True), do_sampling=True))
                                      +
                                      (0.0 if regularization_norm is None else regularization_norm(self.W))
                ) / cv_input_data.shape[0])

            t_total = time.clock() - t_start

            if verbose:
                if len(cost) > 0 and len(cv_cost) > 0:
                    print('Iteration: %s (%s s), train/cv cost: %s / %s' % (n_iter, t_total, cost[-1], cv_cost[-1]))
                elif len(cost) > 0:
                    print('Iteration: %s (%s s), train cost = %s' % (n_iter, t_total, cost[-1]))
                elif len(cv_cost) > 0:
                    print('Iteration: %s (%s s), train cv_cost = %s' % (n_iter, t_total, cv_cost[-1]))

            if n_iter > n_iter_stop_skip:
                if len(cv_cost) > 0:
                    if (1 - stop_threshold) * cv_cost[-1] > min(cv_cost):
                        break
                else:
                    if (1 - stop_threshold) * cost[-1] > min(cost):
                        break
                if len(cost) > 0 and cost[-1] <= min_train_cost:
                    break
                if len(cv_cost) > 0 and cv_cost[-1] <= min_cv_cost:
                    break
                if len(cost) > 1 and abs(cost[-1] - cost[-2]) < tolerance:
                    break



