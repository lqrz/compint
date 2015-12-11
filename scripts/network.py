import numpy as np
import copy
import cPickle as pickle
from scipy.special import expit
from scipy.special import logit
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from sklearn.preprocessing import StandardScaler
from messages import SensorModel, Actions


class Network(object):
    def __init__(self, sensor_ids=None, action_ids=None):
        self.sensor_ids = sensor_ids
        self.action_ids = action_ids

    def get_action(self, sensors):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def crossover(self, partner):
        raise NotImplementedError

    def export(self, filename):
        raise NotImplementedError

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def softmax(x):
        a = np.max(x)
        return np.exp(x-a) / (a + np.log(np.sum(np.exp(x-a))))


class FFNetwork(Network):
    def __init__(self, sensor_ids, action_ids, n_hidden, bias=True):
        super(FFNetwork, self).__init__(sensor_ids=sensor_ids, action_ids=action_ids)
        self.net = buildNetwork(SensorModel.array_length(sensor_ids), n_hidden, 1,
                                hiddenclass=TanhLayer,
                                #outclass=TanhLayer,
                                bias=bias)
        self.scaler_input = None
        self.trainer = None

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            ffn = pickle.load(f)
        self.net = ffn.net
        self.sensor_ids = ffn.sensor_ids
        self.action_ids = ffn.action_ids
        self.scaler_input = ffn.scaler_input
        del ffn

    def get_action(self, sensors):
        x = sensors.get_array(self.sensor_ids)
        if self.scaler_input is not None:
            x = self.scaler_input.transform(x)
        return self.net.activate(x)[0]

    def get_params(self):
        pass

    def train(self, training_files, learningrate=0.01, scaling=True, noise=False, verbose=True):
        print "building dataset..."
        ds = SupervisedDataSet(SensorModel.array_length(self.sensor_ids), 1)
        # read training file line, create sensormodel object, do backprop
        a = None
        s = None
        for logfile in training_files:
            print "loading file", logfile
            with open(logfile) as f:
                for line in f:
                    if line.startswith("Received:"):
                        s = SensorModel(string=line.split(' ', 1)[1])
                    elif line.startswith("Sending:"):
                        a = Actions.from_string(string=line.split(' ', 1)[1])
                    if s is not None and a is not None:
                        ds.addSample(inp=s.get_array(self.sensor_ids), target=a[self.action_ids[0]])
                        if noise:
                            # add the same training sample again but with noise in the sensors
                            s.add_noise()
                            ds.addSample(inp=s.get_array(self.sensor_ids), target=a[self.action_ids[0]])
                        s = None
                        a = None
        print "dataset size:", len(ds)
        if scaling:
            print "scaling dataset"
            self.scaler_input = StandardScaler(with_mean=True, with_std=False).fit(ds.data['input'])
            ds.data['input'] = self.scaler_input.transform(ds.data['input'])
            ds.data['target'] = ds.data['target']
        #self.trainer = BackpropTrainer(self.net, learningrate=learningrate, verbose=verbose)
        self.trainer = RPropMinusTrainer(self.net, verbose=verbose, batchlearning=True)
        print "training network..."
        self.trainer.trainUntilConvergence(dataset=ds, validationProportion=0.25, maxEpochs=10, continueEpochs=2)


class MLP(Network):
    def __init__(self, sensor_ids, action_ids, regression=True):
        super(MLP, self).__init__(sensor_ids=sensor_ids, action_ids=action_ids)
        self.activation_function_h0 = np.tanh
        self.activation_function_h0_string = ''
        self.activation_function_out = Network.linear
        self.activation_function_out_string = ''
        self.hidden_weights = None
        self.hidden_bias = None
        self.out_weights = None
        self.out_bias = None
        self.scaler = None
        self.tau = 0.0
        self.parent_tau = 0.0
        self.eta = [np.array([1.0])] * 4
        self.mutate_dist = np.random.normal  # set to np.random.standard_cauchy for faster evolution
        self.regression = regression

    @staticmethod
    def from_file(file):
        mlp = MLP(sensor_ids=None, action_ids=None)
        mlp.load(file)
        return mlp

    def load(self, pickle_filepath):
        from client import SENSOR_IDXS
        p = pickle.load(open(pickle_filepath, 'rb'))
        try:
            self.hidden_weights = p['h0_W']
            self.eta[0] = np.ones(self.hidden_weights.shape)
            self.hidden_bias = p['h0_b']
            self.eta[1] = np.ones(self.hidden_bias.shape)
        except:
            print 'No hidden layers in pickle file'

        self.out_weights = p['out_W']
        self.eta[2] = np.ones(self.out_weights.shape)
        self.out_bias = p['out_b']
        self.eta[3] = np.ones(self.out_bias.shape)
        self.scaler = p['scaler']
        try:
            self.sensor_ids = p['sensor_fields']
        except KeyError:
            #print "no sensor_fields found in %s, loading default sensor indices" % pickle_filepath
            self.sensor_ids = SENSOR_IDXS
        try:
            self.action_ids = p['action_fields']
        except KeyError:
            #print "no action_fields found in %s, leaving empty" % pickle_filepath
            self.action_ids = ['']
        try:
            self.activation_function_h0 = self.get_activation_function(p['activation_h0'])
            self.activation_function_h0_string = p['activation_h0']
        except:
            # no activation function given in pickle file
            pass
        try:
            self.activation_function_out = self.get_activation_function(p['activation_out'])
            self.activation_function_out_string = p['activation_out']
        except:
            # no activation function given in pickle file
            pass

    def get_activation_function(self, string):
        if string == 'tanh':
            return np.tanh
        elif string == 'sigmoid':
            return expit
        elif string == 'relu':
            return Network.relu
        elif string == 'softmax':
            return Network.softmax
        else:
            return Network.linear

    def set_activation_functions(self):
        self.activation_function_h0 = self.get_activation_function(self.activation_function_h0_string)
        self.activation_function_out = self.get_activation_function(self.activation_function_out_string)

    def get_action(self, sensors):
        x = sensors.get_array(self.sensor_ids)
        x = self.scaler.transform(x)
        if self.hidden_weights:
            hidden_activation = self.activation_function_h0(x.dot(self.hidden_weights) + self.hidden_bias)
        else:
            hidden_activation = x
        y = self.activation_function_out(hidden_activation.dot(self.out_weights) + self.out_bias)

        if not self.regression:
            y = np.argmax(y) - 1 # classes are shifted bt [0,8]
        return y

    def mutate(self, scale_factor=1.0):
        mutant = copy.deepcopy(self)
        mutant.eta[0], mutant.hidden_weights = self.mutate_array(self.hidden_weights, self.eta[0])
        mutant.eta[1], mutant.hidden_bias = self.mutate_array(self.hidden_bias, self.eta[1])
        mutant.eta[2], mutant.out_weights = self.mutate_array(self.out_weights, self.eta[2])
        mutant.eta[3], mutant.out_bias = self.mutate_array(self.out_bias, self.eta[3])
        return mutant

    def mutate_array(self, array, eta=1.0):
        # source: https://www.cs.bham.ac.uk/~xin/papers/published_iproc_sep99.pdf
        tau = np.sqrt(2 * np.sqrt(array.size)) ** -1
        tau_prime = np.sqrt(2 * array.size) ** -1
        eta_prime = eta * np.exp(tau_prime * self.mutate_dist() + tau * self.mutate_dist(size=array.shape))
        return eta_prime, array + eta_prime * self.mutate_dist(size=array.shape)

    def average_arrays(self, array1, array2, ratio=0.5):
        if array1.shape != array2.shape:
            raise ValueError("average arrays must have same shape")
        return (array1 + array2) * ratio

    def crossover(self, partner):
        if not (self.hidden_weights.shape == partner.hidden_weights.shape
                and self.hidden_bias.shape == partner.hidden_bias.shape
                and self.out_weights.shape == partner.out_weights.shape
                and self.out_bias.shape == partner.out_bias.shape):
            raise ValueError("Crossover Partners must have the same network topology")
        child1 = copy.deepcopy(self)
        child1.hidden_weights = self.average_arrays(self.hidden_weights, partner.hidden_weights)
        child1.hidden_bias = self.average_arrays(self.hidden_bias, partner.hidden_bias)
        child1.out_weights = self.average_arrays(self.out_weights, partner.out_weights)
        child1.out_bias = self.average_arrays(self.out_bias, partner.out_bias)
        child2 = child1.mutate()
        return child1, child2


class EchoStateNetwork(Network):
    def __init__(self, sensor_ids, action_ids, n_hidden, bias_value=1.0, leak_rate=0.5):
        super(EchoStateNetwork, self).__init__(sensor_ids=sensor_ids, action_ids=action_ids)
        self.n_hidden = n_hidden
        self.w = None
        self.w_back = None
        self.w_in = None
        self.w_out = None
        self.x_n = None
        self.bias = bias_value
        self.leak_rate = leak_rate
        self.y = np.random.randn(1, 1)
        self.activation = EchoStateNetwork.sigmoid  # np.tanh

    def load(self, w, w_back, w_in, w_out, x_n):
        with open(w) as f:
            self.w = np.array(pickle.load(f))
        with open(w_back) as f:
            self.w_back = np.array(pickle.load(f))
        with open(w_in) as f:
            self.w_in = np.array(pickle.load(f))
        with open(w_out) as f:
            self.w_out = np.array(pickle.load(f))
        with open(x_n) as f:
            self.x_n = np.array(pickle.load(f))

    def random_weights(self, n_input, sparsity=.99, alpha=.5):
        try:
            w = .5 - np.random.randn(self.n_hidden, self.n_hidden)  # TODO: shouldnt w be symmetric?
            w_mask = np.random.randn(self.n_hidden, self.n_hidden) > sparsity
            w = w * w_mask

            max_eigenval = np.max(np.abs(np.linalg.eigvals(w)))

            assert max_eigenval != 0, 'Error! while creating matrix w'

            # TODO: what should i do if the eigenval is complex? drop the img part??
            # if np.imag(max_eigenval) != 0:
            #     print 'Non zero img part',np.imag(max_eigenval)
            #     exit()

            w = w * (alpha / np.real(max_eigenval))
        except AssertionError:
            print 'Found error while initializing matrix w. Trying again.'
            w = self.random_weights(sparsity, alpha)

        self.w = w

        self.w_in = np.random.randn(self.n_hidden, n_input)
        self.w_back = np.random.randn(self.n_hidden, 1)
        self.w_out = np.random.randn(self.n_hidden, 1)
        self.x_n = np.random.randn(self.n_hidden, 1)

    @staticmethod
    def sigmoid(x):
        x[x > 700] = 700
        return expit(x)

    @staticmethod
    def sigmoid_inv(x):
        x[x == 0] = .00000000001
        return logit(x)

    def get_action(self, sensors):
        try:
            x = sensors.get_array(self.sensor_ids)
            if self.w_in is None and self.w_back is None and self.w_out is None:
                self.random_weights(n_input=len(x)+1)
            u = np.matrix(np.concatenate(([self.bias], x), axis=0))
            x_n_update = self.activation(self.w_in.dot(u.T) + self.w.dot(self.x_n) + self.w_back.dot(self.y))
            self.x_n = self.leak_rate * x_n_update + (1 - self.leak_rate) * self.x_n
            self.y = np.squeeze(np.array(self.activation(self.w_out.T.dot(self.x_n)))) * 2 - 1.0
            return self.y
        except Exception as e:
            print "ESN Error:", e
            return 0.0


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.realpath('network.py')) + '/../')

    import logs
    import trained_files
    from client import SENSOR_IDXS
    steer = FFNetwork(sensor_ids=SENSOR_IDXS, action_ids=['steering'], n_hidden=5)
    training_files = [logs.get_file("lau/"+f) for f in os.listdir(logs.get_file("lau/"))][:5]
    print training_files
    steer.train(training_files=training_files, scaling=False)
    with open(trained_files.get_file("pybrain_test.pkl"), 'wb') as f:
        pickle.dump(steer, f)
