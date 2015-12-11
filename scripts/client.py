import socket
import sys
import time
import timeit
import os
import subprocess as sp
import pkg_resources
import random
import psutil
from string import Template
from messages import SensorModel, Actions
from driver import NetworkDriver
from utils import RunningAverage
import trained_files

SENSOR_IDXS = ['angle', 'gear', 'rpm', 'speedX', 'speedY', 'speedZ', 'track', 'trackPos', 'wheelSpinVel', 'z', 'focus']
RECOVERY_LOCK = 10
TRACK_NAMES = {
    'g-track-1': 'road',
    'g-track-2': 'road',
    'g-track-3': 'road',
    # 'aalborg': 'road',
    #'brondehach': 'road',
    # 'corkscrew': 'road',
    'e-track-4': 'road',
    'e-track-6': 'road',
    'a-speedway': 'oval',
    'b-speedway': 'oval',
    'c-speedway': 'oval',
    'd-speedway': 'oval',
    # 'dirt-1': 'dirt',
    # 'dirt-2': 'dirt',
    # 'dirt-3': 'dirt',
    # 'dirt-4': 'dirt',
    # 'dirt-5': 'dirt',
}


class Client:
    def __init__(self, torcs_path=None):
        self.torcs_path = torcs_path
        self.torcs = None
        self.server_address = ('localhost', 3001)
        self.sock = None
        self.init_msg = 'championship2010 1(init -90.0 -60.0 -40.0 -30.0 -25.0 -20.0 -15.0 -10.0 -5.0 0.0 5.0 10.0 15.0 20.0 25.0 30.0 40.0 60.0 90.0)'
        self.auto_recover = True
        self.timeout = None
        self.fnull = open(os.devnull, 'w')

    def configure(self, track_name='g-track-1', laps=1):
        if track_name == 'random':
            track_name = random.choice(TRACK_NAMES.keys())
        with open(pkg_resources.resource_filename("scripts", "quickrace.template.xml")) as race_config:
            tpl = Template(race_config.read())
        try:
            params = dict(trackname=track_name, trackcategory=TRACK_NAMES[track_name],
                          laps=laps, displaymode='results only')
            config = tpl.substitute(params)
            with open(os.path.join(self.torcs_path, "config", "raceman", "quickrace.xml"), 'w') as f:
                f.write(config)
            print "quickrace %i lap(s) on track %s (%s)" % (laps, track_name, TRACK_NAMES[track_name])
            return track_name
        except KeyError:
            print "invalid track name %s. available tracks are %s" % (track_name, TRACK_NAMES.keys())
        except IOError:
            print "configuration file permission denied. please re-run as admin user."

    def connect(self, pipe=None):
        if pipe is None:
            pipe = self.fnull
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.torcs_path is not None:
                #print "starting TORCS subprocess..."
                # torcs_params = ["torcs", "-nofuel", "-nodamage", "-nolaptime", "-noisy"]
                torcs_params = [os.path.join(self.torcs_path, "wtorcs.exe"),
                                "-T", "-nofuel", "-nodamage", "-nolaptime", "-noisy"]
                # self.torcs = sp.Popen(torcs_params, stdout=pipe, stderr=pipe, cwd=self.torcs_path)
                self.torcs = sp.Popen(torcs_params, stdout=pipe, stderr=pipe)
                time.sleep(0.5)
            #print "sending init string to TORCS..."
            self.sock.sendto(self.init_msg, self.server_address)
            data, address = self.sock.recvfrom(128)
            if data.startswith("***identified***"):
                #print "successfully connected to TORCS at host %s:%i." % (address[0], address[1])
                return True
            else:
                print "unexpected reply from TORCS:", data
                return False
        except Exception as e:
            print "Error:", e

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_WR)
            self.sock.close()
            self.sock = None
            p = psutil.Process(pid=self.torcs.pid)
            p.terminate()
            self.torcs = None
        except Exception as e:
            pass
            # print "Client close error:", e

    def race(self, driver):
        """
        Let a driver race in a preconfigured quickrace
        :param driver: a driver object that generates actions based on sensors
        :return: driver fitness value after race
        """
        if not self.connect():
            raise IOError("could not connect to TORCS")
        start_time = timeit.default_timer()
        try:
            print "Start racing..."
            s = None
            lap_times = []
            cur_lap_time = -10.0
            timeout_reached = False
            recovery_lock = 0
            max_speed = 0.0
            avg_speed = RunningAverage()
            driver.prepare()
            while True:
                data = self.sock.recv(2048)
                if data.strip().startswith("("):
                    s = SensorModel(string=data)
                    action = driver.get_action(sensors=s)
                    # save maximum speed for fitness function
                    max_speed = max(max_speed, s['speedX'])
                    avg_speed.add_value(float(s['speedX']))
                    # AUTORECOVERY: if off track, go backwards until back on track and then some more
                    if self.auto_recover and (s.is_off_track() or recovery_lock > 0):
                        action.gear = -1
                        action.accel = 0.4
                        action.clutch = 0.0
                        action.steering = s['angle'] / -2.0
                        if s.is_off_track():
                            recovery_lock = RECOVERY_LOCK
                        else:
                            recovery_lock -= 1
                    self.sock.sendto(str(action), self.server_address)
                    if s['curLapTime'][0] < cur_lap_time:
                        lap_times.append(cur_lap_time)
                        print "lap %i: %0.2f seconds" % (len(lap_times), cur_lap_time)
                    cur_lap_time = s['curLapTime'][0]
                else:
                    if data.startswith("***shutdown***"):
                        if s['curLapTime'][0] > 1:
                            lap_times.append(s['curLapTime'][0])
                        print "--- END OF RACE --- finished at position %i, avg/max speed: %0.2f/%0.2f km/h" % (
                            int(s['racePos']), avg_speed.avg, max_speed)
                        break
                if self.timeout is not None and s['curLapTime'] > self.timeout:
                    print "--- RACE TIMEOUT REACHED ---"
                    timeout_reached = True
                    break
            if s is not None:
                print "lap times:", lap_times
                # print "distance raced:", s['distRaced']
                return driver.compute_fitness(last_sensor=s, lap_times=lap_times, max_speed=max_speed,
                                              average_speed=avg_speed.avg, timeout_reached=timeout_reached)
            else:
                return 0.0
        except KeyboardInterrupt:
            print "Exit client"
        except Exception as e:
            print "Client Error:", e
        finally:
            #print "race call took %0.1f seconds." % (timeit.default_timer() - start_time)
            self.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python client.py torcs_install_path"
        sys.exit(1)

    ndr = NetworkDriver.from_directory(trained_files.get_file("mlp_single_layer"))
    # ndr = NetworkDriver.from_file(trained_files.get_file("best_evo_driver_01.pkl"))

    #c = Client(torcs_path=sys.argv[1])
    #c.configure(track_name='g-track-2', laps=2)
    c = Client(torcs_path=None)  # Use this for GUI test

    c.timeout = 1200
    c.auto_recover = False
    c.race(driver=ndr)
    print "fitness:", ndr.fitness
