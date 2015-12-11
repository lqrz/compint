__author__ = 'lqrz'

import re
import numpy as np
import cPickle as pickle
import sys
import os
import matplotlib.pyplot as plt


def parse_line(line, idxs):
    features = []
    parsed_line = re.findall(r'(?<=\()[^\(]+(?=\))', line)
    # features.extend([float(parsed_line[idx].split(' ')[1:]) for idx in features])
    for feat in np.array(parsed_line)[idxs]:
        features.extend(feat.split()[1:])

    return features

def define_w(sparsity=.99, alpha=.5):
    try:
        w = np.random.randn(n_hidden, n_hidden) #TODO: shouldnt w be symmetric?
        w_mask = np.random.randn(n_hidden, n_hidden) > sparsity
        w = w * w_mask

        max_eigenval = np.linalg.eigvals(w)[0]

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


if __name__=='__main__':
    # log_filepath = os.path.join(os.getcwd(), 'logs\\100lap.log')
    log_filepath = 'logs/100lap.log'
    f_log = open(log_filepath, 'rb')

    # max_samples = 5000000
    # t_0 = 25000 # washout
    n_washout_laps = 10  # nr of washout laps
    max_laps = 20   # nr of laps to consider after washout

    f_plot = True
    f_pickle = False
    # sensor_features = ["angle", "curLapTime", "damage","distFromStart","distRaced","fuel","gear","lastLapTime",
    # 			"opponents","racePos","rpm","speedX","speedY","speedZ","track","trackPos","wheelSpinVel","z","focus"]

    # action_features = ["accel","brake","clutch","gear","steer","meta","focus"]

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

    # sensor_features = ["angle", "gear","rpm","speedX","speedY","speedZ","track","trackPos","wheelSpinVel","z","focus"]
    # action_features = ["steer"]

    #sensor_idxs= [angle_idx, gear_idx, rpm_idx, speedX_idx, speedY_idx, speedZ_idx, track_idx, trackPos_idx,
    #              wheelSpinVel_idx, z_idx, sensor_focus_idx]
    sensor_idxs= [angle_idx, rpm_idx, speedX_idx, speedY_idx, speedZ_idx, track_idx, trackPos_idx, z_idx]
    action_idxs = [action_steer_idx]

    x = []
    d = []
    l = []
    n_lap = 0
    last_dist = 0

    for i,line in enumerate(f_log):

        sys.stdout.write('Processing log line '+str(i+1)+'\r')
        sys.stdout.flush()

        if re.match('^Received: \(', line):

            current_dist = parse_line(line, [distFromStart_idx])[0]
            if current_dist < last_dist:
                n_lap += 1
            last_dist = current_dist
            if n_lap > n_washout_laps+max_laps:
                break

            l.append(max(n_lap,1))
            x.append(parse_line(line, sensor_idxs))

        elif re.match('^Sending: \(', line):
            d.append(parse_line(line, action_idxs))
            # if len(d) > max_samples:
            #     break

    sys.stdout.write('\n')

    x = np.matrix(x, dtype=float)
    sys.stdout.write('x matrix size: %i rows by %i cols\n' % (x.shape[0], x.shape[1]))
    d = np.matrix(d, dtype=float)
    sys.stdout.write('d matrix size: %i rows by %i cols\n' % (d.shape[0], d.shape[1]))

    assert x.shape[0] == d.shape[0], 'X and D have different number of samples!'
    n_samples, n_features = x.shape
    n_hidden = 20
    #n_hidden = np.ceil(n_samples / 10)
    n_out = 1

    print 'Initializing random matrices (includes eigenvalue computation)'
    w_in = np.random.randn(n_hidden, n_features)
    w = define_w(sparsity=.99, alpha=.9)
    w_out = np.random.randn(n_hidden, n_out)
    w_back = np.random.randn(n_hidden, n_out)

    print 'Sampling stage'

    # sampling_iters = min(max_samples, n_samples)

    # m = np.zeros([sampling_iters-t_0,n_hidden])
    # t = np.zeros([sampling_iters-t_0,n_out])
    m = []
    t = []

    x_n = np.zeros([n_hidden,1])

    # for i in range(sampling_iters):
    for i in range(n_samples):

        sys.stdout.write('Sampling lap nr %i \r' % (l[i]))
        sys.stdout.flush()

        x_n = np.tanh(w_in.dot(x[i].T) + w.dot(x_n) + w_back.dot(d[i]))
        assert not np.isnan(x_n).any(), 'Error! NaN found in calculation'

        # if i > t_0:
        if l[i] > n_washout_laps:
            m.append(x_n.T.tolist()[0])
            if np.abs(d[i]) == 1:
                d[i] = d[i] * 0.9999
            tanh_inv = np.arctanh(d[i])[0,0]
            t.append(tanh_inv)

    sys.stdout.write('\n')

    m = np.matrix(m, dtype=float)
    t = np.matrix(t, dtype=float).T

    print 'Weight computation stage (matrix inverse)'
    w_out = np.linalg.pinv(m).dot(t)

    if f_plot:
        trues = []
        preds = []
        n_plot = 500
        y = np.zeros((n_out,1))
        xn_plot = np.zeros((n_hidden,1))
        # steering
        for i in range(n_plot):
            sys.stdout.write('Ploting datapoint %i \r' % (i+1))
            sys.stdout.flush()

            trues.append(d[i,0])
            xn_plot = np.tanh(w_in.dot(x[i].T) + w.dot(xn_plot) + w_back.dot(y))
            y = np.tanh(w_out.T.dot(xn_plot))
            preds.append(y[0,0])
        plt.plot(range(n_plot),trues,label='true')
        plt.plot(range(n_plot),preds,label='pred')
        plt.legend()
        plt.title('Steering')
        plt.savefig('steering_prediction_not_concat.png',dpi=200)

    sys.stdout.write('\n')

    if f_pickle:
        print 'Pickling matrices'
        w_in = w_in.tolist()
        w = w.tolist()
        w_out = w_out.tolist()
        w_back = w_back.tolist()
        x_n = np.asarray(x_n).reshape(-1)
        x_n = x_n.tolist()
        pickle.dump(w_in, open('trained_files/w_in.p', 'wb'))
        pickle.dump(w, open('trained_files/w.p', 'wb'))
        pickle.dump(w_out, open('trained_files/w_out.p', 'wb'))
        pickle.dump(w_back, open('trained_files/w_back.p', 'wb'))
        pickle.dump(x_n, open('trained_files/x_n.p', 'wb'))
        # pickle.dump(w_in, open('../trained_files/w_in.p', 'wb'))
        # pickle.dump(w, open('../trained_files/w.p', 'wb'))
        # pickle.dump(w_out, open('../trained_files/w_out.p', 'wb'))
        # pickle.dump(w_back, open('../trained_files/w_back.p', 'wb'))
        # pickle.dump(x_n, open('../trained_files/x_n.p', 'wb'))

    print 'End'