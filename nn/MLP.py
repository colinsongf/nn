import numpy as np
from Functions import *
from Distances import *
import time
import sys

@np.vectorize
def neuron_local_gain_update(r, c, bonus, penalty, rmin, rmax):
    '''
    update neurons local gain due to condition matrix
    :param r: neurons local gain
    :param c: condition matrix
    :return: new values of neurons local gain
    '''
    return min(bonus + r, rmax) if c else max(penalty * r, rmin)

class MLP:
    # rows are weights of neuron in layer
    # first column is biases, equal to 1
    W = None
    f_list = None

    def __init__(self,
                 input_dimension,
                 layers_structure,
                 activation_functions = None,
                 rng=(lambda n: np.random.normal(0, 0.1, n))
    ):
        """
        Initialization of neural network
        :param input_dimension: input dimension of data without bias
        :param layers_structure: tuple of integers, each value is count of neurons in the layer
        :param activation_functions: tuple of actiovation functions for each of layers, if None then sigmoid is chosen; ndarray -> ndarray
        :param rng: function that returns numpy array of n random numbers
        """
        if activation_functions is not None and len(activation_functions) != len(layers_structure):
            raise Exception("error: activation_functions is not None and len(activation_functions) != len(layers_structure)")
        if activation_functions is None:
            self.f_list = map(lambda x: lambda m: sigmoid(m), range(len(layers_structure)))
        else:
            self.f_list = activation_functions
        self.W = []
        for i in range(len(layers_structure)):
            # neurons in the layer
            neurons = layers_structure[i]
            # weights for biases are added too
            dimension = 1 + (input_dimension if i == 0 else layers_structure[i - 1])
            self.W.append(rng(neurons * dimension).reshape((neurons, dimension)))

    def compute_output(self, input_data, add_bias = True):
        """
        Compute output of network
        :param input_data: ndarray of data, rows are observations
        :param add_bias: add column with biases equals to 1
        """
        if add_bias:
            input_data = np.c_[np.ones(input_data.shape[0]), input_data]
        if input_data.shape[1] != self.W[0].shape[1]:
            raise Exception("error: input_data.shape[1] != self.W[0].shape[1]")
        for i in range(len(self.W)):
            input_data = self.f_list[i](np.dot(input_data, self.W[i].T))
            if i != (len(self.W) - 1):
                input_data = np.c_[np.ones(input_data.shape[0]), input_data]
        return input_data

    def __str__(self):
        return "FeedforwardNN: " + str(self.W[0].shape[1] - 1) + "->" + "->".join(map(lambda m: str(m.shape[0]), self.W))

    def train_backprop(self,
                       input_data, # ndarray
                       output_data, # ndarray
                       learning_rate = 0.1,
                       momentum_rate = 0.9,
                       regularization_rate = 0.1,
                       regularization_norm = None,
                       d_regularization_norm = None,
                       neural_local_gain = (0.05, 0.95, 0.1, 10),  # bonus, penalty, min, max
                       batch_size = None, # None = full batch
                       max_iter = 10000,
                       min_train_cost = np.finfo(float).eps,
                       min_cv_cost = np.finfo(float).eps,
                       n_iter_stop_skip = 10,
                       stop_threshold = 0.05,
                       tolerance = np.finfo(float).eps,
                       goal = xeuclidian, d_goal = d_xeuclidian,
                       d_f_list = None, # list of derivatives of activation functions
                       cv_input_data = None,
                       cv_output_data = None,
                       cv_goal = None,
                       add_bias = True,
                       verbose=True):
        if d_f_list is None:
            d_f_list = map(lambda x: lambda m: sigmoid(m), range(len(self.W)))
        if add_bias:
            input_data = np.c_[np.ones(input_data.shape[0]), input_data]
        if batch_size is None:
            batch_size = input_data.shape[0]
        if neural_local_gain is not None:
            nlg_bonus, nlg_penalty, nlg_min, nlg_max = neural_local_gain
        if cv_goal is None:
            cv_goal = goal
        last_delta_W = [np.zeros(np.prod(m.shape)).reshape(m.shape) for m in self.W]
        # neural local gain: learning rate modifier for each of weights in network
        nlg = [np.ones(np.prod(m.shape)).reshape(m.shape) for m in self.W] if neural_local_gain is not None else None
        cost = []
        cv_cost = []
        do_cv = cv_input_data is not None and cv_output_data is not None
        if do_cv and add_bias:
            cv_input_data = np.c_[np.ones(cv_input_data.shape[0]), cv_input_data]
        for n_iter in range(max_iter):
            t_start = time.clock()
            idx_data = range(input_data.shape[0])
            np.random.shuffle(idx_data)
            idx_batch = range(0, len(idx_data), batch_size)
            for i_in_batch in idx_batch:
                batch_data = input_data[i_in_batch:min(i_in_batch + batch_size, len(idx_data)), :]

                # forward pass
                z = []
                f_z = []
                tmp_data = batch_data
                for i_layer in range(len(self.W)):
                    z.append(np.dot(tmp_data, self.W[i_layer].T))
                    tmp_data = self.f_list[i_layer](z[-1])
                    f_z.append(tmp_data)
                    if i_layer != (len(self.W) - 1):
                        tmp_data = np.c_[np.ones(tmp_data.shape[0]), tmp_data]

                # backward pass
                nabla_W = [np.zeros(np.prod(m.shape)).reshape(m.shape) for m in self.W]
                dE_dz_next = None
                for i_layer in reversed(range(len(self.W))):
                    dE_dz = None
                    if dE_dz_next is None:
                        # output layer
                        dE_dz = d_goal(output_data[i_in_batch:min(i_in_batch + batch_size, len(idx_data)), :], f_z[-1]) * \
                                (1.0 if d_goal == d_cross_entropy else d_f_list[i_layer](z[i_layer]))
                        dE_dz_next = dE_dz
                    else:
                        # any hidden layer
                        dE_dz = self.W[i_layer + 1].T[1:, :].dot(dE_dz_next.T).T * \
                                d_f_list[i_layer](z[i_layer])
                        dE_dz_next = dE_dz
                    layer_input_data = None
                    if i_layer == 0:
                        # FYI: batch data already contains zero column with ones
                        layer_input_data = batch_data
                    else:
                        layer_input_data = np.c_[np.ones(f_z[i_layer - 1].shape[0]), f_z[i_layer - 1]]
                    for i_in_batch_back in range(batch_data.shape[0]):
                        r_dE_dz = np.repeat(dE_dz[i_in_batch_back, :],
                                            self.W[i_layer].shape[1]).reshape(self.W[i_layer].shape)
                        r_layer_input_data = np.repeat(layer_input_data[i_in_batch_back, :],
                                                       self.W[i_layer].shape[0]).reshape(self.W[i_layer].shape, order='F')
                        nabla_W[i_layer] += r_dE_dz * r_layer_input_data
                nabla_W = [w / batch_size for w in nabla_W]

                # update weights
                for i_layer in range(len(self.W)):
                    regularization_penalty = 0.0 if d_regularization_norm is None else d_regularization_norm(self.W[i_layer])
                    if d_regularization_norm is not None:
                        regularization_penalty[:, 0] = 0  # do not penalize bias
                    delta_W = (learning_rate if nlg is None else learning_rate * nlg[i_layer]) * (
                        momentum_rate * last_delta_W[i_layer] +  # momentum: add last delta
                        nabla_W[i_layer] +  # value of gradient
                        (0.0 if d_regularization_norm is None else regularization_rate * regularization_penalty)  # regularization
                    )
                    if nlg is not None:
                        c = delta_W * last_delta_W[i_layer] > 0
                        nlg[i_layer] = neuron_local_gain_update(nlg[i_layer], c, nlg_bonus, nlg_penalty, nlg_min, nlg_max)
                    self.W[i_layer] -= delta_W
                    last_delta_W[i_layer] = delta_W

            # compute cost
            cost.append(np.sum(
                goal(output_data, self.compute_output(input_data, add_bias=False)) +
                (0.0 if regularization_norm is None else np.sum(map(lambda m: regularization_norm(m), self.W)))
            ) / input_data.shape[0])
            if do_cv:
                cv_cost.append(np.sum(
                    cv_goal(cv_output_data, self.compute_output(cv_input_data, add_bias=False)) +
                    (0.0 if regularization_norm is None else np.sum(map(lambda m: regularization_norm(m), self.W)))
                ) / cv_input_data.shape[0])

            t_total = time.clock() - t_start
            if verbose:
                if len(cv_cost) > 0:
                    print('Iteration: %s (%s s), train/cv cost: %s / %s' % (n_iter, t_total, cost[-1], cv_cost[-1]))
                else:
                    print('Iteration: %s (%s s), train cost = %s' % (n_iter, t_total, cost[-1]))

            if n_iter > n_iter_stop_skip:
                if do_cv:
                    if (1 - stop_threshold) * cv_cost[-1] > min(cv_cost):
                        break
                else:
                    if (1 - stop_threshold) * cost[-1] > min(cost):
                        break
                if cost[-1] <= min_train_cost:
                    break
                if cv_cost[-1] <= min_cv_cost:
                    break
                if len(cost) > 1 and abs(cost[-1] - cost[-2]) < tolerance:
                    break

        if do_cv:
            return cost, cv_cost
        else:
            return cost
