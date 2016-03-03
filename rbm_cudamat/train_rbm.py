import numpy as np
import cudamat as cm
import time
import optparse
import matplotlib.pyplot as plt
import os


def save_plot(fname, x, train, cv, xlines=None, min_line=False, dpi=80):
    #plt.figure(figsize=(max(16, min(int(8*len(train)/dpi), 160)), 12), dpi=dpi)
    plt.figure(figsize=(18, 12), dpi=dpi)
    plt.plot(x, train, 'r', label='train')
    plt.plot(x, cv, 'b', label='cv')
    if xlines is not None:
        for x in xlines:
            plt.axvline(x, color='g')
            plt.hlines(np.min(cv_error[:x]), np.argmin(cv_error[:x]), x, color='green', linestyle='dashed')
    if min_line:
        plt.axhline(np.min(cv_error), color='gray', linestyle='dashed')
    plt.legend(loc='upper right')
    plt.title('NN training, cv_min: %0.8f' % np.min(cv_error))
    plt.ylabel('Error')
    plt.xlabel('Epoche')
    plt.savefig(fname)
    plt.clf()
    plt.close()


def harr(dev_m):
    dev_m.copy_to_host()
    return dev_m.numpy_array


def leveldb_batch_generator(db, batch_size=100, mode='v'):
    batch = []
    for key, value in db.RangeIter():
        if mode == 'k':
            batch.append(key)
        elif mode == 'v':
            batch.append(np.frombuffer(value))
        else:
            batch.append((key, np.frombuffer(value)))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-o', dest="output", help='Output file name', default='rbm')
    parser.add_option('--eps', dest="epsilon", type=float, help='Learning rate', default=0.001)
    parser.add_option('--mom', dest="momentum", type=float, help='Momentum rate', default=0.9)
    parser.add_option('--ne', dest="num_epochs", type=int, help='Number of epoches', default=30)
    parser.add_option('--bs', dest="batch_size", type=int, help='Size of the batch', default=32)
    #parser.add_option('--tsr', dest="train_set_ram", type=int, help='Number of train examples loaded to RAM', default=20000)
    parser.add_option('--nv', dest="num_vis", type=int, help='Number of visible units', default=4608)
    parser.add_option('--nh', dest="num_hid", type=int, help='Number of hidden units', default=1024)
    parser.add_option('--gv', dest="gaussian_vis", action='store_true', help='Gaussian visible units', default=False)
    parser.add_option('--mc', dest="minimum_error_change", type=float, help='Minimum error change step', default=0)
    #parser.add_option('--cvr', dest="cv_ratio", type=float, help='Ratio of cross-validation '
    #                                                             'data sampled from input dataset', default=None)
    parser.add_option('--cvmd', dest="cv_stop_factor", type=float, help='Maximum allowed distance between global cv '
                                                                        'error minimum and current value', default=0.1)
    parser.add_option('--do', dest="drop_out", type=float, help='Drop out probability', default=None)
    parser.add_option('--lam', dest='lmbd', type=float, help='Regularization rate', default=None)
    parser.add_option('--rt', dest="reg_type", help='Type of regularization: l1, l2 or en', default='en')
    parser.add_option('--en_lam', dest="en_lmbd", type=float, help='Elastic net lambda', default=0.5)
    parser.add_option('--aef', dest="ada_eps_factor", type=float, help='Adaptive learning rate', default=0.5)
    parser.add_option('--aefm', dest="ada_eps_min", type=float, help='Adaptive learning rate', default=0.00001)
    parser.add_option('--sb', dest="save_best", action='store_true', help='Save best rbm', default=False)
    parser.add_option('-l', dest="load_rbm", help='Load rbm weights', default=None)
    parser.add_option('--ml', dest="memory_limit", type=float, help='GPU memory limit (Mb)', default=2000)
    parser.add_option('-d', dest="debug", action='store_true', help='Debug mode', default=False)
    parser.add_option('--kt', dest="keep_train", action='store_true',
                      help='Do not delete train dataset from RAM (needed for small ds)', default=False)
    parser.add_option('--kv', dest="keep_valid", action='store_true',
                      help='Do not delete valid dataset from RAM (needed for small ds)', default=False)
    options, input_dirs = parser.parse_args()  # first path is to train set, next one is to validation set

    def dlog(s, do_log=options.debug):
        if not do_log:
            return
        print s

    if not os.path.isdir(input_dirs[0]):
        print 'ERROR: Cant find train dir %s' % input_dirs[0]
        exit(0)
    train_path = input_dirs[0] + ('' if input_dirs[0].endswith('/') else '/')

    test_path = None
    if len(input_dirs) > 1:
        if not os.path.isdir(input_dirs[1]):
            print 'ERROR: Cant find valid dir %s' % input_dirs[0]
            exit(0)
        test_path = input_dirs[1] + ('' if input_dirs[1].endswith('/') else '/')

    base_mem_consumption = 0.0

    if options.gaussian_vis:
        print 'Gaussian-Bernoulli mode'
    else:
        print 'Bernoulli-Bernoulli mode'

    # initialize CUDA
    cm.cublas_init()
    base_mem_consumption += 80*1024*1024
    cm.CUDAMatrix.init_random(1)
    base_mem_consumption += 2*1024*1024

    # training parameters
    epsilon = options.epsilon
    momentum = options.momentum
    num_epochs = options.num_epochs
    batch_size = options.batch_size

    # model parameters
    num_vis = options.num_vis
    num_hid = options.num_hid

    # initialize weights
    if options.load_rbm is None:
        w_vh = cm.CUDAMatrix(0.1 * np.random.randn(num_vis, num_hid))
        base_mem_consumption += num_vis*num_hid*4
        w_v = cm.CUDAMatrix(np.zeros((num_vis, 1)))
        base_mem_consumption += num_vis*4
        w_h = cm.CUDAMatrix(-4.*np.ones((num_hid, 1)))
        base_mem_consumption += num_hid*4
    else:
        d = np.load(options.load_rbm)
        w_vh = cm.CUDAMatrix(d['w_vh'])
        base_mem_consumption += np.prod(d['w_vh'].shape)*4
        w_v = cm.CUDAMatrix(d['w_v'])
        base_mem_consumption += np.prod(d['w_v'].shape)*4
        w_h = cm.CUDAMatrix(d['w_h'])
        base_mem_consumption += np.prod(d['w_h'].shape)*4
        del(d)
        print 'INFO: params loaded from ' + options.load_rbm

    epoch_eps_corrected = None
    if options.ada_eps_factor is not None:
        w_vh_min = cm.CUDAMatrix(0.1 * np.random.randn(num_vis, num_hid))
        base_mem_consumption += num_vis*num_hid*4
        w_v_min = cm.CUDAMatrix(np.zeros((num_vis, 1)))
        base_mem_consumption += num_vis*4
        w_h_min = cm.CUDAMatrix(-4.*np.ones((num_hid, 1)))
        base_mem_consumption += num_hid*4
        epoch_eps_corrected = []

    # initialize weight updates
    wu_vh = cm.CUDAMatrix(np.zeros((num_vis, num_hid)))
    base_mem_consumption += num_vis*num_hid*4
    wu_v = cm.CUDAMatrix(np.zeros((num_vis, 1)))
    base_mem_consumption += num_vis*4
    wu_h = cm.CUDAMatrix(np.zeros((num_hid, 1)))
    base_mem_consumption += num_hid*4

    # initialize temporary storage
    v = cm.empty((num_vis, batch_size))
    base_mem_consumption += num_vis*batch_size*4
    h = cm.empty((num_hid, batch_size))
    base_mem_consumption += num_hid*batch_size*4
    r = cm.empty((num_hid, batch_size))
    base_mem_consumption += num_hid*batch_size*4

    # dropout
    if options.drop_out is not None:
        do_h = cm.CUDAMatrix(np.zeros((num_hid, batch_size)))
        base_mem_consumption += num_hid*batch_size*4

    print 'Base memory usage: %0.2f mb' % (base_mem_consumption/(1024**2))
    free_memory = options.memory_limit*1024*1024 - base_mem_consumption
    print 'Free memory: %0.2f mb' % (free_memory/(1024**2))
    if free_memory <= 0:
        print 'ERROR: free memory is negative: %0.2f bytes' % free_memory
        cm.cublas_shutdown()
        exit(0)

    batch_mem_size = num_vis*batch_size*4.0
    print 'One batch memory size (mb): %0.2f' % (batch_mem_size/(1024**2))

    if batch_mem_size > free_memory:
        print 'ERROR: batch_mem_size > free_memory, %0.2f > %0.2f' % (batch_mem_size, free_memory)
        cm.cublas_shutdown()
        exit(0)

    batches_in_free_mem = int(np.floor(free_memory/batch_mem_size))
    print 'Batches in free mem: %i' % batches_in_free_mem

    # control parameters
    train_error = []
    cv_error = []
    dat_train = None
    dat_cv = None

    dlog('Start training')
    for epoch in range(num_epochs):
        start_time = time.time()
        print "Epoch " + str(epoch + 1)
        batch_error_train = []
        batch_error_cv = []

# ---> train set processing
        for fname_tmp in os.listdir(train_path):
            if dat_train is None or not options.keep_train:
                dat_train = np.load(train_path + fname_tmp)['data']
                dlog(' pack lodaed from train: (%s)' % ', '.join(map(lambda i: str(i), dat_train.shape)))
                batch_packs_train = dat_train.shape[0]

            # shuffle data
            np.random.shuffle(dat_train)
            dat_train = dat_train.T

            dlog(' Go through dat_train')
            for batch_pack_inx in range(batch_packs_train):
                dlog('  batch_pack_inx = %i' % batch_pack_inx)
                dat_tmp = dat_train[:, (batch_pack_inx*batch_size*batches_in_free_mem):((batch_pack_inx + 1)*batch_size*batches_in_free_mem)]
                if dat_tmp.shape[1] == 0:
                    break
                try:
                    dev_dat_train = cm.CUDAMatrix(
                        cm.reformat(dat_tmp))
                except Exception as e:
                    print 'CUDAMAT ERROR: ' + e.message
                    cm.cublas_shutdown()
                    exit(0)

                dlog('  dev_dat_train.shape = [%s]' % ', '.join(map(lambda x: str(x), dev_dat_train.shape)))

                num_batches_train = dev_dat_train.shape[1]/batch_size

                for batch in range(num_batches_train):

                    # sample dropout
                    if options.drop_out is not None:
                        do_h.fill_with_rand()
                        do_h.less_than(options.drop_out)

                    # get current minibatch
                    v_true = dev_dat_train.slice(batch*batch_size, (batch + 1)*batch_size)
                    v.assign(v_true)

                    # apply momentum
                    wu_vh.mult(momentum)
                    wu_v.mult(momentum)
                    wu_h.mult(momentum)

                    # positive phase
                    cm.dot(w_vh.T, v, target=h)
                    h.add_col_vec(w_h)
                    h.apply_sigmoid()
                    if options.drop_out is not None:
                        h.mult(do_h)

                    wu_vh.add_dot(v, h.T)
                    wu_v.add_sums(v, axis=1)
                    wu_h.add_sums(h, axis=1)

                    # sample hiddens
                    r.fill_with_rand()
                    r.less_than(h, target=h)

                    # negative phase
                    cm.dot(w_vh, h, target=v)
                    if options.drop_out is not None:
                        v.mult(1/options.drop_out)
                    v.add_col_vec(w_v)
                    if not options.gaussian_vis:
                        v.apply_sigmoid()

                    cm.dot(w_vh.T, v, target=h)
                    h.add_col_vec(w_h)
                    h.apply_sigmoid()
                    if options.drop_out is not None:
                        h.mult(do_h)

                    wu_vh.subtract_dot(v, h.T)
                    wu_v.add_sums(v, axis=1, mult=-1.)
                    wu_h.add_sums(h, axis=1, mult=-1.)

                    # update weights: regularization
                    if options.lmbd is not None:
                        if options.reg_type == 'l1':
                            wu_vh.add_mult(w_vh.sign(), -options.lmbd)
                        elif options.reg_type == 'l2':
                            wu_vh.add_mult(w_vh, -options.lmbd)
                        elif options.reg_type == 'en':
                            wu_vh.add_mult(w_vh.sign(), -options.lmbd * options.en_lmbd)
                            wu_vh.add_mult(w_vh, -options.lmbd * (1 - options.en_lmbd))

                    # update weights: gradients
                    w_vh.add_mult(wu_vh, epsilon/batch_size)
                    w_v.add_mult(wu_v, epsilon/batch_size)
                    w_h.add_mult(wu_h, epsilon/batch_size)

                    # calculate train reconstruction error
                    v.subtract(v_true)
                    batch_error_train.append(v.euclid_norm()**2/(num_vis*batch_size))

                # clear memory
                dev_dat_train.free_device_memory()

            if not options.keep_train:
                del(dat_train)
                dat_train = None
            else:
                dat_train = dat_train.T

# <--- train set processing

# ---> cv set processing

        # calculate cv reconstruction error
        if test_path is not None:
            for fname_tmp in os.listdir(test_path):
                if dat_cv is None or not options.keep_valid:
                    dat_cv = np.load(test_path + fname_tmp)['data']
                    dlog(' pack lodaed from valid: (%s)' % ', '.join(map(lambda i: str(i), dat_cv.shape)))
                    batch_packs_cv = dat_cv.shape[0]
                    dat_cv = dat_cv.T

                dlog(' Go through dat_cv')
                for batch_pack_inx in range(batch_packs_cv):
                    dlog('  batch_pack_inx = %i' % batch_pack_inx)
                    dat_tmp = dat_cv[:, (batch_pack_inx*batch_size*batches_in_free_mem):((batch_pack_inx + 1)*batch_size*batches_in_free_mem)]
                    if dat_tmp.shape[1] == 0:
                        break
                    try:
                        dev_dat_cv = cm.CUDAMatrix(
                            cm.reformat(dat_tmp))
                    except Exception as e:
                        print 'CUDAMAT ERROR: ' + e.message
                        cm.cublas_shutdown()
                        exit(0)

                    dlog('  dev_dat_cv.shape = [%s]' % ', '.join(map(lambda x: str(x), dev_dat_cv.shape)))

                    num_batches_cv = dev_dat_cv.shape[1]/batch_size

                    for batch in range(num_batches_cv):
                        v_true = dev_dat_cv.slice(batch*batch_size, (batch + 1)*batch_size)
                        v.assign(v_true)
                        cm.dot(w_vh.T, v, target=h)
                        h.add_col_vec(w_h)
                        h.apply_sigmoid()
                        r.fill_with_rand()
                        r.less_than(h, target=h)
                        cm.dot(w_vh, h, target=v)
                        v.add_col_vec(w_v)
                        if not options.gaussian_vis:
                            v.apply_sigmoid()
                        v.subtract(v_true)
                        batch_error_cv.append(v.euclid_norm()**2/(num_vis*batch_size))

                    dev_dat_cv.free_device_memory()

                if not options.keep_valid:
                    del(dat_cv)
                    dat_cv = None

# <--- cv set processing

        # reporting
        train_error.append(np.mean(batch_error_train))
        print "  Train MSE: " + str(train_error[-1])
        if test_path is not None:
            cv_error.append(np.mean(batch_error_cv))
            print "  CV MSE: " + str(cv_error[-1])
        print "  Time: " + str(time.time() - start_time)

        # stop conditions
        if len(train_error) > 1:
            if np.abs(train_error[-2] - train_error[-1]) < options.minimum_error_change:
                print 'BREAK: minimum_error_change'
                break

        if test_path is not None and np.min(cv_error) < cv_error[-1] and cv_error[-1]/np.min(cv_error) > options.cv_stop_factor + 1:
            if options.ada_eps_factor is None:
                print 'BREAK: cv_stop_factor, min = %f, cur = %f' % (np.min(cv_error), cv_error[-1])
                break
            elif epsilon < options.ada_eps_min:
                print 'BREAK: ada_eps_min, min = %f, cur = %f' % (options.ada_eps_min, epsilon)
                break
            else:
                w_vh.assign(w_vh_min)
                w_v.assign(w_v_min)
                w_h.assign(w_h_min)
                epsilon = epsilon * options.ada_eps_factor
                epoch_eps_corrected.append(epoch)
                print 'INFO: epsilon is corrected, original = %f, new = %f' % (options.epsilon, epsilon)

        # saving best state
        if len(cv_error) > 1 and np.min(cv_error[:-1]) > cv_error[-1]:
            if options.save_best:
                try:
                    np.savez(options.output + '_min.npz',
                             w_vh=w_vh.asarray(), w_v=w_v.asarray(), w_h=w_h.asarray(),
                             train_error=train_error, cv_error=cv_error)
                    print 'Saved: ' + options.output
                except:
                    print 'ERROR: can\'t save'
                save_plot(options.output + '_plot_min.png', range(epoch + 1), train_error, cv_error, epoch_eps_corrected)

            if options.ada_eps_factor is not None:
                w_vh_min.assign(w_vh)
                w_v_min.assign(w_v)
                w_h_min.assign(w_h)

    try:
        np.savez(options.output + '.npz',
                 w_vh=w_vh.asarray(), w_v=w_v.asarray(), w_h=w_h.asarray(),
                 train_error=train_error, cv_error=cv_error)
        print 'Saved: ' + options.output + '.npz'
    except:
        print 'ERROR: can\'t save'
    save_plot(options.output + '_plot.png', range(epoch + 1), train_error, cv_error, epoch_eps_corrected, min_line=True)

    cm.cublas_shutdown()
