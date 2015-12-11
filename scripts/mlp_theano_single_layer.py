from IPython.kernel.comm.manager import with_output

__author__ = 'root'
from dataset import read_logs
import theano.tensor as T
import theano
import numpy as np
import cPickle as cp
import time
import sys

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.warn_float64='ignore'
theano.config.floatX='float64'

class Hidden_layer_classification:

    def __init__(self, n_in, n_hidden, activation_f, W=None, b=None):

        # this is an heuristically set initialization of weights W
        lim = np.sqrt(6./(n_in+n_hidden))
        if activation_f == theano.tensor.nnet.sigmoid:
            lim *= 4

        if W:
            self.W = theano.shared(value=W, name='W', borrow=True)
        else:
            self.W = theano.shared(value=np.random.uniform(-lim,lim,(n_in,n_hidden)).astype(dtype=theano.config.floatX), name='W', borrow=True)

        if b:
            self.b = theano.shared(value=b, name='b', borrow=True)
        else:
            self.b = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='b', borrow=True)

        self.activation_f = activation_f

        # self.a = self.activation_f(T.dot(x, self.W) + self.b)
        # self.y_pred = T.argmax(self.a, axis=1)  # prediction (argmax)

        self.params = [self.W, self.b]

    def forward_pass(self, x):
        z = T.dot(x, self.W) + self.b
        self.a = self.activation_f(z) if self.activation_f else z

        return self.a

    def prediction(self, x):
        z = T.dot(x, self.W) + self.b
        a = self.activation_f(z) if self.activation_f else z

        return T.argmax(a)

    def cost(self, y_true):
        return -T.mean(T.log(self.a)[T.arange(y_true.shape[0]), y_true])
        # return T.mean(T.sqr(self.a[:,0] - y_true))

    def cost2(self, trues):
        return -T.mean(T.log(self.y_pred))
        # return self.a[:,0] - trues

    def errors(self, x, y_true):
        y_pred = self.prediction(x)
        # return T.mean(T.sqr(y_pred[:,0] - y_true))
        return T.sum(T.neq(y_pred, y_true))

class Hidden_layer:

    def __init__(self, n_in, n_hidden, activation_f, W=None, b=None):

        # this is an heuristically set initialization of weights W
        lim = np.sqrt(6./(n_in+n_hidden))
        if activation_f == theano.tensor.nnet.sigmoid:
            lim *= 4

        if W:
            self.W = theano.shared(value=W, name='W', borrow=True)
        else:
            self.W = theano.shared(value=np.random.uniform(-lim,lim,(n_in,n_hidden)).astype(dtype=theano.config.floatX), name='W', borrow=True)

        if b:
            self.b = theano.shared(value=b, name='b', borrow=True)
        else:
            self.b = theano.shared(value=np.zeros((n_hidden,), dtype=theano.config.floatX), name='b', borrow=True)

        self.activation_f = activation_f

        # self.a = self.activation_f(T.dot(x, self.W) + self.b)
        # self.y_pred = T.argmax(self.a, axis=1)  # prediction (argmax)

        self.params = [self.W, self.b]

    def forward_pass(self, x):
        z = T.dot(x, self.W) + self.b
        self.a = self.activation_f(z) if self.activation_f else z

        return self.a

    def prediction(self, x):
        z = T.dot(x, self.W) + self.b
        a = self.activation_f(z) if self.activation_f else z

        return a

    def cost(self, y_true):
        # return -T.mean(T.log(self.a)[T.arange(y_true.shape[0]), y_true])
        return T.mean(T.sqr(self.a[:,0] - y_true))

    def cost2(self, trues):
        # return -T.mean(T.log(self.y_pred))
        return self.a[:,0] - trues

    def errors(self, x, y_true):
        y_pred = self.prediction(x)
        return T.mean(T.sqr(y_pred[:,0] - y_true))


class MLP:
    def __init__(self, x, n_in, n_out, out_activation_f, regression=True,
                 h0_W=None, h0_b=None, out_W=None, out_b=None):

        if regression:
            self.out_layer = Hidden_layer(n_in, n_out, out_activation_f, W=out_W, b=out_b)
        else:
            self.out_layer = Hidden_layer_classification(n_in, n_out, out_activation_f, W=out_W, b=out_b)
        self.out_layer.forward_pass(x)    # compute hidden layer activation

        self.cost = self.out_layer.cost

        self.params = self.out_layer.params
        # self.learning_rate = learning_rate

        self.L1 = (abs(self.out_layer.W).sum())

        self.L2 = (self.out_layer.W ** 2).sum()

    def predict(self, x):
        a = self.out_layer.prediction(x)

        return a

    def errors(self, x, y):
        return self.out_layer.errors(x, y)


class MLPTrainer():

    def __init__(self, dataset_path, sensor_idxs, action_idxs, max_laps, n_out, out_activation_f,
                 regression=True):

        x_train, y_train, x_valid, y_valid, scaler = read_logs(dataset_path, sensor_idxs, action_idxs,  max_laps,
                                                               noise=False, shuffled=True,
                                                               scale=True, valid_prop=.2)
        # indices must be integers for classification
        if regression:
            dtype = theano.config.floatX
            self.y_train = y_train.astype(dtype=theano.config.floatX)
            self.y_valid = y_valid.astype(dtype=theano.config.floatX)

        else:
            dtype = int
            self.y_train = y_train.astype(dtype=dtype)+1
            self.y_valid = y_valid.astype(dtype=dtype)+1

        self.x_train = x_train.astype(dtype=theano.config.floatX)
        self.x_valid = x_valid.astype(dtype=theano.config.floatX)
        self.sensor_idxs = sensor_idxs
        self.action_idxs = action_idxs
        self.max_laps = max_laps
        self.scaler = scaler
        self.network = None
        self.n_out = n_out
        self.out_activation_f = out_activation_f
        self.out_activation_f_name = get_activation_function_name(out_activation_f)
        self.regression = regression

    def train(self, learning_rate=0.01, batch_size=600, max_epochs=100,
              L1_reg=0.001, L2_reg=0.01, error_thresh=0.0001):

        train_x = theano.shared(value=np.array(self.x_train, dtype=theano.config.floatX), name='train_x', borrow=True)

        if self.regression:
            train_y = theano.shared(value=np.array(self.y_train, dtype=theano.config.floatX), name='train_y', borrow=True)
        else:
            train_y = theano.shared(value=np.array(self.y_train, dtype='int64'), name='train_y', borrow=True)

        x = T.matrix(name='x', dtype=theano.config.floatX)
        if self.regression:
            y = T.vector(name='y', dtype=theano.config.floatX)
        else:
            y = T.vector(name='y', dtype='int64')

        n_in = self.x_train.shape[1]

        self.network = MLP(x, n_in, self.n_out, self.out_activation_f, regression=self.regression)

        idx = T.lscalar()  # index to a [mini]batch
        n_train_batches = train_x.get_value().shape[0] / batch_size

        cost = self.network.cost(y) + L1_reg*self.network.L1 + L2_reg*self.network.L2

        # adagrad
        accumulated_grad = []
        for param in self.network.params:
            eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        grads = [T.grad(cost, param) for param in self.network.params]

        updates = []
        for param, grad, accum_grad in zip(self.network.params, grads, accumulated_grad):
            # accum = T.cast(accum_grad + T.sqr(grad), dtype=theano.config.floatX)
            accum = accum_grad + T.sqr(grad)
            updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
            updates.append((accum_grad, accum))

        train_model = theano.function(inputs=[idx],
                                      outputs=cost,
                                      updates=updates,
                                      givens={
                                          x: train_x[idx*batch_size:(idx+1)*batch_size],
                                          y: train_y[idx*batch_size: (idx+1)*batch_size]
                                      })

        valid_errors_f = theano.function(inputs=[x,y], outputs=self.network.errors(x,y))
        epoch = 0
        train_errors = []
        valid_errors = []
        cost = error_thresh
        while(epoch < max_epochs):
            for batch_idx in xrange(n_train_batches):
                # z,pred = h0_activ(batch_idx)
                # print out_activ(z)
                # print cost_act(batch_idx)
                cost = train_model(batch_idx)
                # print 'Train batch index %d cost %f' % (batch_idx, cost)
            train_error = valid_errors_f(self.x_train, self.y_train)
            valid_error = valid_errors_f(self.x_valid, self.y_valid)
            print 'Epoch %d Cost: %f Train errors: %f Valid errors: %f' % (epoch, cost, train_error, valid_error)
            last_cost = cost
            epoch += 1

        print '...Prediction'
        pred_f = theano.function(inputs=[x], outputs=self.network.predict(x))
        n_predict = 20
        for i in range(n_predict):
            a = pred_f(np.matrix(self.x_valid[i], dtype=theano.config.floatX))
        #     y_pred = T.cast(res, 'int32')
        #     a = T.cast(a, theano.config.floatX)
        #     print 'Input: ', np.matrix(self.x_valid[i], dtype=theano.config.floatX)
            print 'True: %f Pred: %f' % (self.y_valid[i], np.squeeze(a))
        #     # print 'Activation: ', T.cast(a, theano.config.floatX).eval().shape

    def save_pickles(self, pickle_outpath, sensor_fields, action_fields):
        print '...Pickling files'
        pickles = dict()
        pickles['sensor_fields'] = sensor_fields
        pickles['action_fields'] = action_fields
        pickles['scaler'] = self.scaler
        pickles['scaler_means'] = self.scaler.mean_
        pickles['scaler_stds'] = self.scaler.std_
        pickles['out_W'] = self.network.out_layer.W.get_value()
        pickles['out_b'] = self.network.out_layer.b.get_value()

        # cp.dump(pickles, open(pickle_outpath+'_'.join([str(self.network.n_hidden),str(time.time())])+'.pkl', 'wb'))
        cp.dump(pickles, open(pickle_outpath+'.pkl', 'wb'))

    def write_matrix_values(self, output_file, matrix):
        for i in range(matrix.shape[0]):
            np.savetxt(output_file, matrix[i,:], delimiter=' ', newline=' ', fmt='%.18e')
            output_file.write('\n')
        output_file.write('\n\n')

        return True

    def save_text(self, text_outpath, sensor_fields):

        w_out_hidden = np.array(self.network.out_layer.W.get_value(), dtype=float)
        w_out_hidden = np.vstack((w_out_hidden, self.network.out_layer.b.get_value()))

        filename = text_outpath+'.txt'
        with open(filename, 'w') as f:

            # write sensor names
            f.write(' '.join(sensor_fields)+'\n\n')

            # write activation function descriptions
            f.write(self.out_activation_f_name+'\n\n')

            # write scaler data
            np.savetxt(f, (self.scaler.mean_, self.scaler.std_),
                       delimiter=' ', newline='\n\n', fmt='%.18e')

            # write matrix weights
            self.write_matrix_values(f, w_out_hidden)

        return True

def get_activation_function_name(activation_function):
    name = None
    if activation_function == T.tanh:
        name = 'tanh'
    elif activation_function == T.nnet.sigmoid:
        name = 'sigmoid'
    elif activation_function == T.nnet.relu:
        name = 'relu'
    elif activation_function == T.nnet.softmax:
        name = 'softmax'
    elif not activation_function:
        name = 'linear'

    assert name, 'Error while identifying activation function'

    return name


def acceleration_mlp(input_path, pickle_outpath):

    # sensor_fields = ['angle', 'gear', 'rpm', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel',
    #            'z']

    sensor_fields = ['angle', 'rpm', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel', 'z']

    action_fields = ['accel']

    max_laps = 20
    n_out = 1

    out_activation_f = T.nnet.relu

    trainer = MLPTrainer(input_path, sensor_fields, action_fields, max_laps, n_out, out_activation_f)
    trainer.train(batch_size=600, max_epochs=600, L1_reg=0.001, L2_reg=0.001)
    trainer.save_pickles(pickle_outpath, sensor_fields, action_fields)
    trainer.save_text(pickle_outpath, sensor_fields)

def steering_mlp(input_path, pickle_outpath):
    sensor_fields = ['angle', 'gear', 'rpm', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel',
               # 'z', 'focus']
               'z']
    # action_fields = ['steer', 'accel', 'brake']
    action_fields = ['steer']

    max_laps = 20
    n_out = 1

    out_activation_f = T.tanh

    trainer = MLPTrainer(input_path, sensor_fields, action_fields, max_laps, n_out, out_activation_f)
    trainer.train(batch_size=600, max_epochs=1000, L1_reg=0.001, L2_reg=0.001)
    trainer.save_pickles(pickle_outpath, sensor_fields, action_fields)
    trainer.save_text(pickle_outpath, sensor_fields)

def brake_mlp(input_path, pickle_outpath):
    # sensor_idxs = [angle_idx, gear_idx, rpm_idx, speedX_idx, speedY_idx, speedZ_idx, track_idx, trackPos_idx,
    #                wheelSpinVel_idx, z_idx, sensor_focus_idx]
    sensor_fields = ['angle', 'rpm', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel',
               # 'z', 'focus']
               'z']
    action_fields = ['brake']

    max_laps = 20
    n_out = 1

    out_activation_f = T.tanh

    trainer = MLPTrainer(input_path, sensor_fields, action_fields, max_laps, n_out, out_activation_f)
    trainer.train(batch_size=600, max_epochs=600, L1_reg=0.001, L2_reg=0.001)
    trainer.save_pickles(pickle_outpath, sensor_fields, action_fields)
    trainer.save_text(pickle_outpath, sensor_fields)

def gear_mlp(input_path, pickle_outpath):
    # sensor_idxs = [angle_idx, gear_idx, rpm_idx, speedX_idx, speedY_idx, speedZ_idx, track_idx, trackPos_idx,
    #                wheelSpinVel_idx, z_idx, sensor_focus_idx]
    sensor_fields = ['gear', 'rpm', 'speedX', 'speedY', 'speedZ']
    action_fields = ['gear']

    max_laps = 20
    n_out = 9

    out_activation_f = T.nnet.softmax

    trainer = MLPTrainer(input_path, sensor_fields, action_fields, max_laps, n_out, out_activation_f,
                         regression=False)
    trainer.train(batch_size=600, max_epochs=2000, L1_reg=0.001, L2_reg=0.001)
    trainer.save_pickles(pickle_outpath, sensor_fields, action_fields)
    trainer.save_text(pickle_outpath, sensor_fields)


if __name__=='__main__':
    # input_path = 'logs/berniw_aalborg/'
    input_path = 'logs/berniw_aalborg/'
    pickle_outpath = 'trained_files/theano_mlp/'

    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        pickle_outpath = sys.argv[2]
    elif len(sys.argv) > 1:
        print 'Error in call'
        exit()

    identifier = str(time.time())

    print '...Training steering mlp'
    # steering_mlp(input_path, pickle_outpath+'steering'+identifier)

    print '...Training acceleration mlp'
    # acceleration_mlp(input_path, pickle_outpath+'acceleration'+identifier)

    print '...Training braking mlp'
    brake_mlp(input_path, pickle_outpath+'brake'+identifier)

    print '...Training gear mlp'
    # gear_mlp(input_path, pickle_outpath+'gear'+identifier)

    print 'End'