__author__ = 'lqrz'

import numpy as np
import re
import sys
from pybrain.structure import TanhLayer
from pybrain.structure import LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer

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


def train(log_filepath, sensor_idxs, action_idxs, max_laps):
    f_log = open(log_filepath, 'rb')

    x = []
    d = []
    l = []
    n_lap = 0
    last_dist = 0

    for i,line in enumerate(f_log):
        sys.stdout.write('Processing log line '+str(i+1)+'\r')
        sys.stdout.flush()

        # if re.match('^LAP\s5.*', line):
        #     print 'lines: ',len(x)

        if re.match('^Received: \(', line):

            current_dist = parse_line(line, [distFromStart_idx])[0]
            if float(current_dist) - float(last_dist) < -1000:
                n_lap += 1
            last_dist = current_dist
            if n_lap > max_laps:
                break

            l.append(max(n_lap,1))

            x.append(parse_line(line, sensor_idxs))
        elif re.match('^Sending: \(', line):
            d.append(parse_line(line, action_idxs))

    sys.stdout.write('\n')

    x = np.array(x, dtype=float)
    d = np.array(d, dtype=float)

    assert x.shape[0] == d.shape[0], 'X and D have different number of samples!'

    n_hidden = 100
    ds = SupervisedDataSet(x.shape[1], d.shape[1])
    ds.setField('input', x)
    ds.setField('target', d)

    net = buildNetwork(x.shape[1], n_hidden, d.shape[1], bias=True, outputbias=True, hiddenclass=TanhLayer, outclass=TanhLayer)
    trainer = BackpropTrainer(net, ds, learningrate=0.01, verbose=True)
    trainer.trainUntilConvergence(validationProportion=0.15, maxEpochs=10, continueEpochs=10)
    # trainer.trainOnDataset(ds, 200, verbose=True)
    params = net.params
    w_in = net.params[:n_hidden*x.shape[1]].reshape((n_hidden,x.shape[1]))
    b_in = net.params[n_hidden*x.shape[1]:n_hidden*x.shape[1]+n_hidden]
    w_out = net.params[n_hidden*x.shape[1]+n_hidden:n_hidden*x.shape[1]+n_hidden+n_hidden*d.shape[1]].reshape((d.shape[1],n_hidden))
    b_out = net.params[n_hidden*x.shape[1]+n_hidden+n_hidden*d.shape[1]:]

    for mod in net.modules:
        if mod._name == 'out':
            continue
        params = net.connections[mod][0].params
        if mod._name == 'in':
            w_in = params.reshape(x.shape[1],n_hidden)
        if mod._name == 'hidden0':
            w_out = params
        if mod._name =='bias':
            b_in = params
        if mod._name =='outputbias':
            b_out = params
    for i in range(10):
        true_val = d[i]
        man_pred = w_out.T.dot(np.tanh(w_in.T.dot(x[i]) + b))
        aut_pred = net.activate(x[i])


if __name__=='__main__':

    log_filepath = 'logs/100lap.log'

    f_train_steering = True
    f_train_accel = True

    if f_train_steering:
        sensor_idxs= [angle_idx, trackPos_idx, speedX_idx]
        action_idxs = [action_steer_idx]
        train(log_filepath, sensor_idxs, action_idxs, max_laps=1)