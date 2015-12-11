import re
import numpy as np

PATTERN = re.compile(r'(?<=\()[^\(]+(?=\))')

SENSOR_LENGTHS = {
    'angle': 1,
    'curLapTime': 1,
    'damage': 1,
    'distFromStart': 1,
    'distRaced': 1,
    'focus': 5,
    'fuel': 1,
    'gear': 1,
    'lastLapTime': 1,
    'opponents': 36,
    'racePos': 1,
    'rpm': 1,
    'speedX': 1,
    'speedY': 1,
    'speedZ': 1,
    'track': 19,
    'trackPos': 1,
    'wheelSpinVel': 4,
    'z': 1
}

NOISE_SENSORS = ['angle', 'focus', 'rpm', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel', 'z']

class Message(object):
    def parse_string(self, string):
        d = dict()
        parsed_line = re.findall(PATTERN, string)
        for sensor in parsed_line:
            k, v = sensor.split(' ', 1)
            d[k] = np.array(v.split()).astype(np.float)
        return d

    def get_array(self, ids):
        return np.concatenate([self[sid] for sid in ids])


class SensorModel(Message):
    def __init__(self, string):
        super(SensorModel, self).__init__()
        self.string = string
        self.sensors = self.parse_string(string)

    def __getitem__(self, item):
        return self.sensors.get(item, np.array([0.0,]))

    def __str__(self):
        return self.string

    def is_off_track(self):
        return np.all(self.sensors['track'] <= -1.0)

    def add_noise(self, sigma=0.02):
        for k, v in self.sensors.items():
            if k in NOISE_SENSORS:
                try:
                    v_new = np.random.normal(v, np.abs(v*sigma))
                except:
                    v_new = np.random.normal(v, 0.00001)
                self.sensors[k] = np.array(v_new)
                if len(self.sensors[k].shape) == 0:
                    self.sensors[k] = np.array([self.sensors[k]])

    @staticmethod
    def array_length(sensor_ids):
        return sum([SENSOR_LENGTHS.get(s, 1) for s in sensor_ids])



class Actions(Message):
    def __init__(self, accel=0.0, brake=0.0, clutch=0.0, gear=0, steering=0.0, focus=360, meta=0):
        super(Actions, self).__init__()
        self.accel = max(0.0, min(1.0, accel))
        self.brake = max(0.0, min(1.0, brake))
        self.clutch = max(0.0, min(1.0, clutch))
        self.gear = max(-1, min(6, gear))
        self.steering = max(-1.0, min(1.0, steering))
        self.focus = focus
        self.meta = meta

    def __getitem__(self, item):
        try:
            return self.__getattribute__(item)
        except:
            return 0.0

    def __str__(self):
        return '(accel %f) (brake %f) (clutch %f) (gear %i) (steer %f) (meta %i) (focus %i)' % (
                self.accel, self.brake, self.clutch, self.gear, self.steering, self.meta, self.focus)

    @staticmethod
    def from_string(string):
        m = Message()
        d = m.parse_string(string)
        a = Actions(d.get('accel', 0.), d.get('brake', 0.), d.get('clutch', 0.), d.get('gear', 0),
                    d.get('steering', 0.), d.get('focus', 360), d.get('meta', 0))
        return a


if __name__ == '__main__':
    sm = SensorModel(
        string='(angle 0.00969284)(curLapTime -0.982)(damage 0)(distFromStart 2032.56)(distRaced 0)(fuel 94)(gear 0)(lastLapTime 0)(opponents 176.307 200.481 214.049 185.952 257.951 211.456 206.264 223.249 210.815 175.645 201.462 215.976 184.841 212.032 183.762 190.385 192.251 186.109 187.283 170.886 197.641 224.287 186.507 217.084 192.956 216.853 196.624 198.666 181.698 167.123 184.679 171.805 206.981 199.336 223.793 178.59)(racePos 1)(rpm 942.478)(speedX -0.000612781)(speedY 0.00284858)(speedZ -0.000184746)(track 4.46775 6.31817 8.95216 10.8851 18.5271 148.539 143.368 79.1084 57.346 40.863 28.8228 22.2623 25.8085 22.2586 15.8062 19.0807 15.1714 13.4133 8.22151)(trackPos 0.333666)(wheelSpinVel 0 0 0 0)(z 0.345609)(focus -1 -1 -1 -1 -1)')
    sm.add_noise()
    sm.is_off_track()
    print sm['track', 'angle']
    #'(accel 1.0) (brake 0.0) (clutch 0.0) (gear 1) (steer -0.2000771164894104) (meta 0) (focus 360)'
