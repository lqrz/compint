import copy
from messages import Actions
import pickle as cp
import sys
import glob
import os
import numpy as np
from network import MLP


class Driver(object):
    def __init__(self):
        self.fitness = -1.0
        self.clutch_value = 0.0
        self.clutch_dec = 0.01
        self.gear_value = 1
        self.gear_lock = 0
        self.comment = ''

    def prepare(self):
        self.gear_value = 1
        self.gear_lock = 0
        self.clutch_value = 0.0

    def get_action(self, sensors):
        raise NotImplementedError

    def export(self, filename):
        raise NotImplementedError

    def update_gear(self, sensors):
        if self.gear_lock:
            self.gear_lock -= 1
        else:
            if self.gear_value > 1 and sensors['rpm'] < 3000:
                self.gear_value -= 1
                self.clutch_value = 0.5
                self.gear_lock = 40
            elif self.gear_value < 7 and sensors['rpm'] > 6000:
                self.gear_value += 1
                self.clutch_value = 0.5
                self.gear_lock = 40
        return self.gear_value

    def update_clutch(self):
        if self.clutch_value > 0:
            self.clutch_value -= self.clutch_dec
            self.clutch_dec *= 1.3
        if self.clutch_value < 0:
            self.clutch_value = 0.0
            self.clutch_dec = 0.01
        return self.clutch_value

    def compute_fitness(self, last_sensor, lap_times, max_speed, average_speed, timeout_reached=False):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def crossover(self, partner):
        raise NotImplementedError


class NetworkDriver(Driver):
    def __init__(self, max_speed=None):
        super(NetworkDriver, self).__init__()
        self.steering = None
        self.accel = None
        self.gear = None
        self.brake = None
        self.max_speed = max_speed
        self.default_accel = 0.3
        self.brake_factor = np.random.uniform()

    def get_networks(self):
        return [net for net in [self.steering, self.accel, self.gear, self.brake] if net is not None]

    def get_action(self, sensors):
        a = Actions(accel=self.default_accel)
        if self.steering is not None:
            a.steering = self.steering.get_action(sensors)

        if self.accel is not None:
            a.accel = self.accel.get_action(sensors)
        elif self.max_speed is not None and abs(sensors['speedX']) > self.max_speed:
            a.accel = 0.05

        if self.gear is not None:
            a.gear = self.gear.get_action(sensors)
        else:
            a.gear = self.update_gear(sensors)
            a.clutch = self.update_clutch()

        if self.brake is not None:
            a.brake = self.brake.get_action(sensors) * self.brake_factor
        return a

    def compute_fitness(self, last_sensor, lap_times, max_speed, average_speed, timeout_reached=False):
        self.fitness = float((1.0 / (last_sensor['racePos'])) *
                             max_speed * 0.01 *
                             average_speed * 0.1)
        if timeout_reached:
            self.fitness = self.fitness * float(last_sensor['distRaced'] * 0.001)
        if len(lap_times) > 0:
            self.fitness = self.fitness * len(lap_times) / sum(lap_times)
        return self.fitness

    def mutate(self, **kwargs):
        mutant = copy.deepcopy(self)
        for net in mutant.get_networks():
            net = net.mutate(**kwargs)
        mutant.mutate_brake_factor()
        return mutant

    def mutate_brake_factor(self):
        self.brake_factor = max(1.0, abs(self.brake_factor + np.random.normal(scale=0.5)))

    def crossover(self, partner):
        child1, child2 = copy.deepcopy(self), copy.deepcopy(partner)
        if child1.steering is not None and child2.steering is not None:
            steering1, steering2 = child1.steering.crossover(partner=child2.steering)
            child1.steering = steering1
            child2.steering = steering2
        if child1.accel is not None and child2.accel is not None:
            accel1, accel2 = child1.accel.crossover(partner=child2.accel)
            child1.accel = accel1
            child2.accel = accel2
        if child1.brake is not None and child2.brake is not None:
            brake1, brake2 = child1.brake.crossover(partner=child2.brake)
            child1.brake = brake1
            child2.brake = brake2
        if child1.gear is not None and child2.gear is not None:
            gear1, gear2 = child1.gear.crossover(partner=child2.gear)
            child1.gear = gear1
            child2.gear = gear2
        # reset fitness of children
        child1.fitness = child2.fitness = -1.0
        child1.mutate_brake_factor()
        child2.mutate_brake_factor()
        return child1, child2

    def __repr__(self):
        return "NetworkDriver with fitness %f (comment: %s)" % (self.fitness, str(self.comment))

    def export(self, filename):
        for net in self.get_networks():
            try:
                net.activation_function_h0 = None
                net.activation_function_out = None
            except:
                pass
        try:
            with open(filename, 'wb') as f:
                cp.dump(self, f)
        except Exception as e:
            print >> sys.stderr, "NetworkDriver could not be exported:", e
        for net in self.get_networks():
            try:
                net.set_activation_functions()
            except:
                pass
        return True

    @staticmethod
    def from_file(filename):
        with open(filename, 'rb') as f:
            ndr = cp.load(f)
        for net in ndr.get_networks():
            try:
                net.set_activation_functions()
            except:
                pass
        return ndr

    @staticmethod
    def from_directory(path):
        # check for .pkl files in the given directoy
        ndr = NetworkDriver()
        file_list = glob.glob(os.path.join(path, "*.pkl"))
        for pickle_file in file_list:
            filename = os.path.basename(pickle_file)
            if filename.startswith('accel') and ndr.accel is None:
                ndr.accel = MLP.from_file(pickle_file)
                print "loaded accel. ",
            elif filename.startswith('steering') and ndr.steering is None:
                ndr.steering = MLP.from_file(pickle_file)
                print "loaded steering. ",
            elif filename.startswith('brake') and ndr.brake is None:
                ndr.brake = MLP.from_file(pickle_file)
                print "loaded brake. ",
            elif filename.startswith('gear') and ndr.gear is None:
                ndr.gear = MLP.from_file(pickle_file)
                ndr.gear.regression = False
                print "loaded gear. ",
        print ""
        return ndr
