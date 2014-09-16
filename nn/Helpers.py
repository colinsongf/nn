import numpy as np

@np.vectorize
def neuron_local_gain_update(r, c, bonus, penalty, rmin, rmax):
    '''
    update neurons local gain due to condition matrix
    :param r: neurons local gain
    :param c: condition matrix
    :return: new values of neurons local gain
    '''
    return min(bonus + r, rmax) if c else max(penalty * r, rmin)