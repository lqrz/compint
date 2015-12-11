__author__ = 'lqrz'
import sys
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from os import walk
import codecs

SENSOR_IDXS = ['angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced', 'fuel', 'gear', 'lastLapTime',
               'opponents', 'racePos', 'rpm', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel',
               'z', 'focus']

ACTION_IDXS = ['accel', 'brake', 'clutch', 'gear', 'steer', 'meta', 'focus']

sensor_dict = dict(zip(SENSOR_IDXS, range(len(SENSOR_IDXS))))
action_dict = dict(zip(ACTION_IDXS, range(len(ACTION_IDXS))))


def parse_line(line, idxs):
    features = []
    parsed_line = re.findall(r'(?<=\()[^\(]+(?=\))', line)
    # features.extend([float(parsed_line[idx].split(' ')[1:]) for idx in features])
    for feat in np.array(parsed_line)[idxs]:
        features.extend(feat.split()[1:])

    return features

def read_logs(path, sensor_fields, action_fields, max_laps,
              shuffled=False, noise=False, scale=False, valid_prop=None):
    x = []
    d = []

    file_names = None
    for (_,_,fnames) in walk(path):
        file_names = map(lambda x: path+x, fnames)

    assert file_names, 'Couldnt retrieve log files. Bad path: '+path

    for f_name in file_names:
        n_lap = 0
        last_dist = 0
        f_out = codecs.open(f_name, 'rb', encoding='utf-8')
        for i,line in enumerate(f_out):

            sys.stdout.write('Processing log '+f_name+' line '+str(i+1)+'\r')
            sys.stdout.flush()

            if re.match('^Received: \(', line):
                sensor_idxs = [sensor_dict['distFromStart']]
                current_dist = parse_line(line, sensor_idxs)[0]
                if float(current_dist) - float(last_dist) < -1000:
                    n_lap += 1
                last_dist = current_dist
                if n_lap > max_laps:
                    break

                sensor_idxs = [sensor_dict[field] for field in sensor_fields]
                data = parse_line(line, sensor_idxs)

                x.append(data)

                if noise:
                    x.append(np.random.normal(loc=data,
                                              scale=np.max(np.abs(np.array(data,dtype=float)*.02), .000001)).tolist())
                    x.append(np.random.normal(loc=data,
                                              scale=np.max(np.abs(np.array(data,dtype=float)*.02), .000001)).tolist())

            elif re.match('^Sending: \(', line):

                action_idxs = [action_dict[field] for field in action_fields]
                data = parse_line(line, action_idxs)

                d.append(data)

                if noise:
                    d.append(data)
                    d.append(data)

        sys.stdout.write('\n')
        sys.stdout.flush()

    if shuffled:
        print '...Shuffling dataset'
        x = np.matrix(x)
        d = np.matrix(d)
        shuf = np.concatenate((x,d), axis=1)
        np.random.shuffle(shuf)
        x = shuf[:,:-1]
        d = shuf[:,-1]

    x = np.array(x, dtype=float)
    d = np.array(d, dtype=float)
    d = np.squeeze(np.array(d, dtype=float))

    assert x.shape[0] == d.shape[0], 'X and D have different number of samples!'

    scaler = None
    if scale:
        print '...Scaling dataset'
        scaler = StandardScaler(with_mean=True, with_std=True).fit(x)
        x = scaler.transform(x)

    x_train = None
    y_train = None
    x_valid = None
    y_valid = None
    if valid_prop:
        print '...Splitting dataset'
        valid_lines = valid_prop * x.shape[0]
        x_train = x[:-valid_lines]
        y_train = d[:-valid_lines]
        x_valid = x[-valid_lines:]
        y_valid = d[-valid_lines:]

    return x_train, y_train, x_valid, y_valid, scaler

if __name__ == '__main__':
    file_names = ['logs/12lap.log']
    sensor_idxs= [angle_idx, trackPos_idx, speedX_idx]
    action_idxs = [action_steer_idx]
    max_laps = 3

    x_train, y_train, x_valid, y_valid, scaler = read_logs(file_names, sensor_idxs, action_idxs, max_laps,
                                                           noise=True, shuffled=True)

    print 'End'