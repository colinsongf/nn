import numpy as np
from nn.MLP import MLP, sigmoid, d_sigmoid, d_identity, identity, tanh, d_tanh
from nn.MLP import xeuclidian, d_xeuclidian, log_Bernoulli_likelihood, d_log_Bernoulli_likelihood
from nn.Norms import l2, d_l2
from scipy import misc
import os
import re

if __name__ == '__main__':


    train_input = np.empty((0, 841))
    cv_input = np.empty((0, 841))
    test_input = np.empty((0, 841))
    train_output = np.empty((0, 26))
    cv_output = np.empty((0, 26))
    test_output = np.empty((0, 26))

    for f in os.listdir('./../data/big_alphabet_29x29/'):
        v = np.array(misc.imread('./../data/big_alphabet_29x29/' + f, flatten=True)).flatten() / 255.0
        parts = re.split('[-\.]', f)
        i = int(parts[1])
        o = np.zeros(26)
        o[i] = 1.0
        if len(parts) == 5 and parts[2] in ['7', '8']:
            cv_input = np.vstack([cv_input, v])
            cv_output = np.vstack([cv_output, o])
        elif len(parts) == 5 and parts[2] in ['5', '6']:
            test_input = np.vstack([cv_input, v])
            test_output = np.vstack([cv_output, o])
        else:
            train_input = np.vstack([train_input, v])
            train_output = np.vstack([train_output, o])


    nn = MLP(841, (100, 26),
                       activation_functions=[sigmoid, sigmoid],
                       rng=(lambda n: np.random.normal(0, 0.01, n)))
    train_cost, cv_cost = \
        nn.train_backprop(train_input, train_output,
                          d_f_list=[d_sigmoid, d_sigmoid],
                          goal=log_Bernoulli_likelihood,
                          d_goal=d_log_Bernoulli_likelihood,
                          batch_size=None,
                          max_iter=2500,
                          learning_rate=0.1,
                          momentum_rate=0.9,
                          neural_local_gain=(0.005, 0.995, 0.001, 1000),
                          stop_threshold=0.05,
                          cv_input_data=cv_input,
                          cv_output_data=cv_output,
                          #regularization_rate=0.1,
                          #regularization_norm=l2,
                          #d_regularization_norm=d_l2
                          verbose=True
                          )

    t = np.argmax(train_output, axis=1)
    y = np.argmax(nn.compute_output(train_input), axis=1)

    print('%s / %s' % (sum(t == y), train_output.shape[0]))

    t = np.argmax(test_output, axis=1)
    y = np.argmax(nn.compute_output(test_input), axis=1)

    print('%s / %s' % (sum(t == y), test_output.shape[0]))


