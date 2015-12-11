__author__ = 'lqrz'

import re
import numpy as np
import cPickle as pickle
import sys
import os
import itertools as it
import operator
import timeit
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.special import logit

# sensor feature indexes
angle_idx = 0
curLapTime_idx = 1
damage_idx = 2
distFromStart_idx = 3
distRaced_idx = 4
fuel_idx = 5
gear_idx = 6
lastLapTime_idx = 7
opponents_idx = 8
racePos_idx = 9
rpm_idx = 10
speedX_idx = 11
speedY_idx = 12
speedZ_idx = 13
track_idx = 14
trackPos_idx = 15
wheelSpinVel_idx = 16
z_idx = 17
sensor_focus_idx = 18

# action feature indexes
action_accel_idx = 0
action_brake_idx = 1
action_clutch_idx = 2
action_gear_idx = 3
action_steer_idx = 4
action_meta_idx = 5
action_focus_idx = 6

def parse_line(line, idxs):
    features = []
    parsed_line = re.findall(r'(?<=\()[^\(]+(?=\))', line)
    # features.extend([float(parsed_line[idx].split(' ')[1:]) for idx in features])
    for feat in np.array(parsed_line)[idxs]:
        features.extend(feat.split()[1:])

    return features

def define_w(n_hidden, sparsity=.99, alpha=.5):
    try:
        w = .5-np.random.randn(n_hidden, n_hidden) #TODO: shouldnt w be symmetric?
        w_mask = np.random.randn(n_hidden, n_hidden) > sparsity
        w = w * w_mask

        assert w[w!=0].any(), 'Error! while creating matrix w'

        max_eigenval = np.max(np.abs(np.linalg.eigvals(w)))

        assert max_eigenval != 0, 'Error! while creating matrix w'

        #TODO: what should i do if the eigenval is complex? drop the img part??
        # if np.imag(max_eigenval) != 0:
        #     print 'Non zero img part',np.imag(max_eigenval)
        #     exit()

        w = w*(alpha/np.real(max_eigenval))
    except AssertionError:
        print 'Found error while initializing matrix w. Trying again.'
        w = define_w(sparsity, alpha)

    return w

def load_data(log_filepath, sensor_idxs, action_idxs):
    f_log = open(log_filepath, 'rb')
    lap_end = []
    x = []
    d = []
    n_lap = 0
    last_dist = 0
    sample_nr = 0

    for i,line in enumerate(f_log):
        sys.stdout.write('Processing log line '+str(i+1)+'\r')
        sys.stdout.flush()

        # if re.match('^LAP\s5.*', line):
        #     print 'lines: ',len(x)

        if re.match('^Received: \(', line):

            current_dist = parse_line(line, [distFromStart_idx])[0]
            if float(current_dist) - float(last_dist) < -1000:
                n_lap += 1
                lap_end.append(sample_nr)
            last_dist = current_dist
            # if n_lap > n_washout_laps+max_laps:
            #    break
            x.append(parse_line(line, sensor_idxs))
        elif re.match('^Sending: \(', line):
            d.append(parse_line(line, action_idxs))
            sample_nr += 1

    # lap_end.append(i)
    sys.stdout.write('\n')

    x = np.matrix(x, dtype=float)
    sys.stdout.write('x matrix size: %i rows by %i cols\n' % (x.shape[0], x.shape[1]))
    d = np.matrix(d, dtype=float)
    sys.stdout.write('d matrix size: %i rows by %i cols\n' % (d.shape[0], d.shape[1]))

    assert x.shape[0] == d.shape[0], 'X and D have different number of samples!'
    return x, d, n_lap, lap_end


def train_esn(x, d, w_sparsity,
              w_alpha, n_hidden, alpha_reg, leak_rate, bias_value, activation_f, activation_inv_f,
              f_pickle, f_plot, out_dir, performance_samples, n_out, washout_threshold):
    '''
    n_washout_laps = 30  # nr of washout laps
    max_laps = 30   # nr of laps to consider after washout
    w_sparsity = .99    # sparsity of reservoir
    w_alpha = 1    # alpha param of reservoir
    n_hidden = 22  # size of reservoir
    alpha_reg = 1**-1   # regularization param
    leak_rate = .45 # leaking rate
    bias_value = 1  # bias value to add
    :return:
    '''

    n_samples, n_features = x.shape

    print 'Initializing random matrices (includes eigenvalue computation)'

    if bias_value:
        w_in = .5-np.random.randn(n_hidden, n_features+1)  # add bias
    else:
        w_in = .5-np.random.randn(n_hidden, n_features)  # do not add bias

    w = define_w(n_hidden, sparsity=w_sparsity, alpha=w_alpha)
    w_out = .5-np.random.randn(n_hidden+n_features, n_out)
    w_back = .5-np.random.randn(n_hidden, n_out)

    print 'Sampling stage'

    # sampling_iters = min(max_samples, n_samples)

    # m = np.zeros([sampling_iters-t_0,n_hidden+n_features])
    # t = np.zeros([sampling_iters-t_0,n_out])
    m = []
    t = []

    x_n = np.zeros([n_hidden,1])
    d_n = np.zeros([n_out,1])

    # for i in range(sampling_iters):
    for i in range(n_samples):

        if bias_value:
            u = np.matrix(np.concatenate(([bias_value], np.array(x)[i]), axis=0))     # add bias to input
        else:
            u = np.matrix(np.array(x)[i])     # do not add bias to input

        x_n_update = activation_f(w_in.dot(u.T) + w.dot(x_n) + w_back.dot(d_n))
        d_n = d[i].T
        assert not np.isnan(x_n_update).any(), 'Error! NaN found in calculation'
        x_n = leak_rate * x_n_update + (1-leak_rate) * x_n
        assert not np.isnan(x_n).any(), 'Error! NaN found in calculation'

        if i > washout_threshold:
            m.append(np.concatenate((x_n, x[i].T), axis=0).T.tolist()[0])

            if np.abs(d_n) == 1:
                d_n = d_n * .9999999999999999

            tanh_inv = np.squeeze(np.array(activation_inv_f(d_n)))
            assert not np.isnan(tanh_inv) or tanh_inv == float('inf') or tanh_inv == float('-inf'),\
                'ERROR! Incorrect inverse computation'
            t.append(tanh_inv)

    m = np.matrix(m, dtype=float)
    t = np.matrix(t, dtype=float)

    print 'Weight computation stage (matrix inverse)'
    if alpha_reg > 0:
        '''
        Ridge Regression
        '''
        # w_out = np.linalg.pinv(m + np.eye(m.shape[0], m.shape[1], dtype=float)*(alpha_reg**2)).dot(t)
        w_out = np.linalg.pinv(m.T.dot(m) + alpha_reg * np.identity(m.shape[1])).dot(m.T.dot(t.T))
    else:
        '''
        Moore Penrose Pseudo Inverse Method
        (K + N) << number of samples!
        '''
        w_out = np.linalg.pinv(m).dot(t.T)

    assert not np.isnan(w_out).any(), 'Error! NaN found while computing w_out'

    if f_plot or performance_samples:
        n_plot = min(5000, len(d))
        trues = np.zeros(n_plot)
        preds = np.zeros(n_plot)
        # y = np.zeros((n_out,1))
        # xn_plot = np.zeros((n_hidden,1))
        y = d_n  # prediction 49
        xn_plot = x_n   # ready for prediction 49
        for i in range(n_plot):
            sys.stdout.flush()
            trues[i] = d[i,0]
            if bias_value:
                u = np.matrix(np.concatenate(([bias_value], np.array(x)[i]), axis=0))  # add bias to input
            else:
                u = np.matrix(np.array(x)[i])  # do not add bias to input

            xn_plot_update = activation_f(w_in.dot(u.T) + w.dot(xn_plot) + w_back.dot(y))
            assert not np.isnan(xn_plot_update).any(), 'Error! NaN found while predicting'
            xn_plot = leak_rate * xn_plot_update + (1-leak_rate) * xn_plot
            assert not np.isnan(xn_plot).any(), 'Error! NaN found while predicting'
            y = np.squeeze(np.array(activation_f(w_out.T.dot(np.concatenate((xn_plot, x[i].T), axis=0)))))
            assert not np.isnan(y) or y == float('inf') or y == float('-inf'), 'ERROR! Incorrect prediction output'
            preds[i] = y

    if f_plot:
        plt.plot(range(n_plot),trues,label='true')
        plt.plot(range(n_plot),preds,label='pred')
        plt.legend()
        plt.title('Approximation')
        plt.ylim((-1,1))
        plt.savefig(out_dir+'/approximation.png',dpi=200)

    performance = 0.0
    if performance_samples:
        performance = np.sqrt(np.sum((trues - preds)**2))

    if f_pickle:
        print 'Pickling matrices'
        w_in = w_in.tolist()
        w = w.tolist()
        w_out = w_out.tolist()
        w_back = w_back.tolist()
        x_n = np.asarray(x_n).reshape(-1)
        x_n = x_n.tolist()
        pickle.dump(w_in, open(out_dir+'/w_in.p', 'wb'))
        pickle.dump(w, open(out_dir+'/w.p', 'wb'))
        pickle.dump(w_out, open(out_dir+'/w_out.p', 'wb'))
        pickle.dump(w_back, open(out_dir+'/w_back.p', 'wb'))
        pickle.dump(x_n, open(out_dir+'/x_n.p', 'wb'))

    print 'End'

    return performance


def sigmoid(x):
    x[x>700] = 700

    return expit(x)

def sigmoid_inv(x):
    x[x==0] = .00000000001

    return logit(x)


# def map_sigmoid(x):
#     return np.matrix(map(sigmoid,np.array(x)[:,0])).T


if __name__=='__main__':
    print "ESN Concatenated - cwd:", os.getcwd()
    '''
    ESN hyperparams:
                    n_washout_laps
                    max_laps
                    leak_rate
                    n_hidden
                    sparsity
                    alpha
                    alpha_reg
                    bias_value
    '''

    log_filepath = 'logs/simpleDriver/12lap.log'

    # sensor_features = ["angle", "curLapTime", "damage","distFromStart","distRaced","fuel","gear","lastLapTime",
    # 			"opponents","racePos","rpm","speedX","speedY","speedZ","track","trackPos","wheelSpinVel","z","focus"]

    # action_features = ["accel","brake","clutch","gear","steer","meta","focus"]

    # sensor_features = ["angle", "gear","rpm","speedX","speedY","speedZ","track","trackPos","wheelSpinVel","z","focus"]
    # action_features = ["steer"]

    #sensor_idxs= [angle_idx, gear_idx, rpm_idx, speedX_idx, speedY_idx, speedZ_idx, track_idx, trackPos_idx,
    #              wheelSpinVel_idx, z_idx, sensor_focus_idx]

    # sensor_idxs= [angle_idx, track_idx, z_idx, sensor_focus_idx]

    f_train_steering = True
    f_train_accel = True
    
    sensor_idxs= [angle_idx, trackPos_idx, speedX_idx]
    action_idxs = [action_steer_idx]

    x, d, n_lap, lap_ends = load_data(log_filepath, sensor_idxs, action_idxs)
    
    steering_params = {
                       "w_sparsity": [0.99],
                       "w_alpha": [0.5],
                       "n_hidden": [5],
                       "alpha_reg": [10**-2],
                       "leak_rate": [1],
                       "washout_threshold": [lap_ends[n_washout-1] for n_washout in [11]
                                             if len(lap_ends) > n_washout and n_washout > 0]
                       }
    
    varNames = sorted(steering_params)
    combinations = [dict(zip(varNames, prod)) for prod in it.product(*(steering_params[varName] for varName in varNames))]
    performance_samples = 5000
    steering_performance = {}
    accel_performance = {}
    start = timeit.default_timer()
    
    for i, params in enumerate(combinations):
        print "\ntraining parameter set %i of %i:\n%s" % (i+1, len(combinations), params)
        p = params # backup because params are changed for print
        if f_train_steering:
            s_performance = train_esn(x=x, d=d[:,0], bias_value=None,
                                    activation_f=np.tanh, activation_inv_f=np.arctanh, f_pickle=True, f_plot=True,
                                    out_dir='trained_files/steering', performance_samples=performance_samples, n_out=1, **params)
            print "steering error:", s_performance
            params['washout_threshold'] = lap_ends.index(params['washout_threshold']) + 1
            steering_performance[str(params)] = s_performance
    
        if f_train_accel:
            a_performance = train_esn(x=x, d=d[:,1], bias_value=1,
                                    activation_f=np.tanh, activation_inv_f=np.arctanh, f_pickle=True, f_plot=False,
                                    out_dir='trained_files/steering', performance_samples=performance_samples, n_out=1, **p)
            print "accel error:", a_performance
            accel_performance[str(params)] = a_performance
            
    with open("trained_files/steering_error.pkl", 'wb') as f:
        pickle.dump(steering_performance, f, protocol=0)
        
    with open("trained_files/accel_error.pkl", 'wb') as f:
        pickle.dump(accel_performance, f, protocol=0)
        
    print "FINISHED after %f seconds." % (timeit.default_timer() - start)
                                                                                                          
    if f_train_steering:
        print "Best steering parameters: %s" % min(steering_performance.iteritems(), key=operator.itemgetter(1))[0]
    if f_train_accel:
        print "Best acceleration parameters: %s" % min(accel_performance.iteritems(), key=operator.itemgetter(1))[0]