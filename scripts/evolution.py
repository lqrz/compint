import sys
import cPickle as cp
import timeit
import datetime
import random
from client import Client
from driver import NetworkDriver
import trained_files


class Evolution(object):
    def __init__(self, torcs_client):
        self.torcs = torcs_client
        self.population = []
        self.previous_population = []
        self.fitness_history = []
        self.pop_fitness_history = []
        self.track = 'random'
        self.tracks_used = []
        self.laps = 3
        self.output_file = trained_files.get_file("best_evo_driver.pkl")
        # set this to true to have fitness values comparable across different tracks
        self.normalize_fitness=False

    def sort_population(self):
        # sort population by decreasing driver fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def normalize_population_fitness(self, population=None):
        if population is None:
            population = self.population
        f = [driver.fitness for driver in population]
        for driver in population:
            driver.fitness = (driver.fitness - min(f)) / (max(f) - min(f))

    def evaluate(self, population=None):
        self.tracks_used.append(c.configure(track_name=self.track, laps=self.laps))
        if population is None:
            population = self.population
        for d, driver in enumerate(population):
            fit = c.race(driver=driver)
            print "> Driver %i/%i reached fitness %f" % (d+1, len(population), fit)
        if self.normalize_fitness:
            self.normalize_population_fitness()
        print "evaluated pop fitness: ", [driver.fitness for driver in population]
        self.sort_population()
        self.fitness_history.extend([d.fitness for d in self.population])

    def sum_fitness(self):
        return sum([d.fitness for d in self.population])

    def evolve(self, iteration_index):
        # temp copy of current population
        current_population = list(self.population)
        # union population of current and previous (children and parents)
        self.population.extend(self.previous_population)
        # take the best drivers of the union to create a new population (while retaining the very best driver)
        self.sort_population()
        new_population = list()
        for n in range(0, len(current_population)-1, 2):
            # crossover of two parents returns two children
            children = self.population[n].crossover(partner=self.population[n + 1])
            children[0].comment = children[1].comment = iteration_index
            new_population.extend(children)
        # compute fitness of new children
        print "> Evaluating new children..."
        self.evaluate(new_population)
        # now we keep only the best of all the drivers that are currently alive
        self.population.extend(new_population)
        print "> Total number of drivers alive:", len(self.population)
        # bad drivers die (2/3 of the union)
        self.sort_population()
        del self.population[len(current_population):]
        # now parents
        self.previous_population = current_population

    def run(self, iterations=10):
        start = timeit.default_timer()
        try:
            if len(self.population) < 2:
                raise ValueError("Population size %i too small" % len(self.population))
            # evaluate initial population
            self.evaluate(self.population)
            for m in range(iterations):
                print "\n## Evolution Iteration %02i ##" % (m+1)
                self.evolve(m)
                print "> Total population fitness: %0.4f" % self.sum_fitness()
                print "Population birth times (sorted):", [driver.comment for driver in self.population]
                self.pop_fitness_history.append(self.sum_fitness())
        except Exception as e:
            print >> sys.stderr, "EVOLUTION ERROR:", e
        finally:
            print "Population fitness history:", self.pop_fitness_history
            self.population[0].export(self.output_file)
            print "\n### FINISHED EVOLUTION AFTER %0.2f SECONDS ###" % (timeit.default_timer() - start)
            return self.population[0]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python evolution.py torcs_install_path"
        sys.exit(1)
    c = Client(torcs_path=sys.argv[1])

    # CONFIGURATION #
    POPULATION_SIZE = 10
    BATCHES = 5
    EVO_ITERATIONS = 3
    TRACK = 'random'
    LAPS = 2
    NORMALIZED_FITNESS = True
    c.timeout = LAPS * 500
    c.auto_recover = True
    #################

    trained_pop = []
    trained_pop.append(NetworkDriver.from_directory(trained_files.get_file("Human")))
    trained_pop.append(NetworkDriver.from_directory(trained_files.get_file("SimpleDriver")))
    trained_pop.append(NetworkDriver.from_directory(trained_files.get_file("SimpleExample2")))
    trained_pop.append(NetworkDriver.from_directory(trained_files.get_file("bot1")))
    trained_pop.append(NetworkDriver.from_directory(trained_files.get_file("bot2")))

    evo = Evolution(torcs_client=c)
    global_population_fitness_history = []
    global_tracks = []
    start = timeit.default_timer()
    pop = list(trained_pop)
    while len(pop) < POPULATION_SIZE:
        new_driver = random.choice(trained_pop).mutate()
        pop.append(new_driver)
    evo.laps = LAPS
    evo.track = TRACK
    evo.normalize_fitness = NORMALIZED_FITNESS
    for driver in pop:
        driver.comment = -1
    evo.population = pop
    print "# start: %s #" % datetime.datetime.now().isoformat()
    for i in range(BATCHES):
        print "\n############################\n##### EVOLUTION RUN %02i #####\n############################" % (i+1)
        evo.output_file = trained_files.get_file("best_evo_driver_%02i.pkl" % (i+1))
        evo.run(iterations=EVO_ITERATIONS)
        # take best two drivers from this batch as seed driver for the next batch
        pop = evo.population[:2]
        global_population_fitness_history.append(evo.pop_fitness_history)
        global_tracks.append(evo.tracks_used)
    try:
        with open(trained_files.get_file("global_population_fitness_history.pkl"), 'wb') as f:
            cp.dump(dict(global_population_fitness_history=global_population_fitness_history,
                         global_tracks=global_tracks), f)
    except:
        print "could not save global pop fit hist"
    finally:
        print "\n##### FINISHED BATCH EVOLUTION AFTER %0.2f SECONDS #####" % (timeit.default_timer() - start)
        print "# end: %s #" % datetime.datetime.now().isoformat()
