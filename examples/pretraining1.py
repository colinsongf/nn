import sys
sys.path.append('./../nn/')

import numpy as np
from pandas import read_csv
from nn.nn.MLP import MLP, sigmoid, d_sigmoid, d_identity, identity, tanh, d_tanh, mcrmse, xeuclidian, d_xeuclidian, hamming, euclidian
from nn.nn.RBM import RBM
from nn.nn.Norms import l2, d_l2, l1, d_l1
import dill
import multiprocessing
import os
from scipy import misc
import re


if __name__ == '__main__':
    train_input = np.empty((0, 841))
    #cv_input = np.empty((0, 841))
    #test_input = np.empty((0, 841))
    #train_output = np.empty((0, 26))
    #cv_output = np.empty((0, 26))
    #test_output = np.empty((0, 26))

    for f in os.listdir('./../data/big_alphabet_29x29/'):
        v = np.abs((np.array(misc.imread('./../data/big_alphabet_29x29/' + f, flatten=True)).flatten() / 255.0) - 1)
        parts = re.split('[-\.]', f)
        i = int(parts[1])
        o = np.zeros(26)
        o[i] = 1.0
        train_input = np.vstack([train_input, v])
        # if len(parts) == 5 and parts[2] in ['7', '8']:
        #     cv_input = np.vstack([cv_input, v])
        #     cv_output = np.vstack([cv_output, o])
        # elif len(parts) == 5 and parts[2] in ['5', '6']:
        #     test_input = np.vstack([cv_input, v])
        #     test_output = np.vstack([cv_output, o])
        # else:
        #     train_input = np.vstack([train_input, v])
        #     train_output = np.vstack([train_output, o])

    rbm = RBM(841, 100,
              rng=(lambda n: np.random.normal(0, 0.001, n)))

    print(rbm)

    rbm.train(train_input,
              cd_k=1,
              learning_rate=0.01,
              momentum_rate=0.9,
              max_iter=1000,
              batch_size=10,
              n_iter_stop_skip=10,
              goal=hamming,
              #cv_input_data=cv_input,
              stop_threshold=0.15,
              #neural_local_gain=(0.05, 0.95, 0.01, 100),
              #regularization_rate=0.1,
              #regularization_norm=l1,
              #d_regularization_norm=d_l1,
    )