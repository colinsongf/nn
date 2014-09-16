import numpy as np
from pandas import read_csv
from nn.MLP import MLP, sigmoid, d_sigmoid, d_identity, identity, tanh, d_tanh, mcrmse, xeuclidian, d_xeuclidian, hamming, euclidian
from nn.RBM import RBM
from nn.Norms import l2, d_l2, l1, d_l1


if __name__ == '__main__':
    df = read_csv('./../data/africa-soil/training.csv')
    x = df.as_matrix(columns=df.columns[1:3595])
    x[:, -1] = (x[:, -1] == 'Topsoil') * 1.0
    x = x.astype(float)
    y = df.as_matrix(columns=df.columns[3595:])
    y = y.astype(float)

    # standartizing
    x = (x - np.repeat(x.mean(axis=0), x.shape[0]).reshape((x.shape[0], x.mean(axis=0).shape[0]), order='F')) / \
        np.sqrt(np.repeat(x.var(axis=0), x.shape[0]).reshape((x.shape[0], x.mean(axis=0).shape[0]), order='F'))

    idx_train = list(np.random.choice(range(x.shape[0]), size=int(round(0.8 * x.shape[0]))))
    idx_cv = list(set(range(x.shape[0])) - set(idx_train))

    rbm = RBM(x.shape[1], 100,
              rng=(lambda n: np.random.normal(0, 0.001, n)),
              mode='gaus-bin',
              use_biases=True)

    print(rbm)

    rbm.train(x[idx_train, :],
              cd_k=1,
              learning_rate=0.001,
              momentum_rate=0.9,
              max_iter=1000,
              batch_size=20,
              n_iter_stop_skip=10,
              goal=euclidian,
              #cv_input_data=cv_input,
              stop_threshold=0.15,
              #neural_local_gain=(0.05, 0.95, 0.01, 100),
              regularization_rate=0.1,
              #regularization_norm=l1,
              d_regularization_norm=d_l1,
    )
