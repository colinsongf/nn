import numpy as np
from pandas import read_csv
from nn.MLP import MLP, sigmoid, d_sigmoid, d_identity, identity, tanh, d_tanh
from nn.Norms import l2, d_l2
import dill


if __name__ == '__main__':

    df = read_csv('./../data/africa-soil/training.csv')
    x = df.as_matrix(columns=df.columns[1:3595])
    x[:, -1] = (x[:, -1] == 'Topsoil') * 1.0
    x = x.astype(float)
    y = df.as_matrix(columns=df.columns[3595:])
    y = y.astype(float)

    idx_train = list(np.random.choice(range(x.shape[0]), size=int(round(0.8 * x.shape[0]))))
    idx_cv = list(set(range(x.shape[0])) - set(idx_train))

    nn = MLP(3594, (50, 5),
                       activation_functions=[tanh, identity],
                       rng=(lambda n: np.random.normal(0, 0.01, n)))
    train_cost, cv_cost = \
        nn.train_backprop(x[idx_train, :], y[idx_train, :],
                          d_f_list=[d_tanh, d_identity],
                          batch_size=None,
                          max_iter=1000,
                          learning_rate=0.001,
                          momentum_rate=0.9,
                          neural_local_gain=(0.0005, 0.9995, 0.001, 1000),
                          stop_threshold=0.05,
                          cv_input_data=x[idx_cv, :],
                          cv_output_data=y[idx_cv, :],
                          #regularization_rate=0.1,
                          #regularization_norm=l2,
                          #d_regularization_norm=d_l2
                          verbose=True
                          )