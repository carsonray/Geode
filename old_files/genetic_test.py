import old_core1 as net
import genetic as gen

import numpy as np
import matplotlib.pyplot as plt

class GenSys(net.System):
    def gradient_test(self, dense, exp, poprange, popsize, epochs):
        """Runs genetic algorithms on a random gradient and finds the highest point"""
        # Sets up plots with comparison on y-axis and epochs on x-axis
        # Leaves extra spot for comparitive stat results
        #f, axis = plt.subplots(len(self.map), epochs)
        #self.axis = axis

        # Creates random gradient with a size of one more than the population range (so there is no overflow)
        gradient = gen.rand_grad(0, 1, dense, exp, size=(np.array(poprange) + 1))
        self.gradient = gradient

        # Sets up algorithm for first genAlg and transfers population to other algorithms
        pop = self.map[0].setup(poprange, popsize)
        
        # Runs setups for all algorithms
        for system in self.map[1:]:
            system.setup(poprange, popsize)
        
        # Runs genetic alogrithms in parallel
        self.recur((pop,), epochs)

class GradGen(gen.GenAlg):
    def grad_assess(self, points):
        """Assesses population points based off of gradient"""
        # Rounds all points
        points = np.round(points)
        points = points.astype(int)
        
        # Finds layer of system
        layer = self.parent.map.index(self)
        # Displays population
        #self.parent.axis[layer, self.epoch].imshow(self.parent.gradient, cmap=plt.get_cmap("plasma"))
        #self.parent.axis[layer, self.epoch].scatter(*points.swapaxes(0,1), c="#000000")
        
        # Reshapes points so that fancy indexing per axis can be used
        values = self.parent.gradient[tuple(points.swapaxes(0,1))]

        # Returns points and values
        return points, values

def comp_metrics(genSys, vars, dense=0.1, exp=0.8, poprange=(100,100), popsize=20, epochs=30, tests=20):
    maxes = np.zeros(len(genSys.map))

    speeds = maxes.copy()

    for test in range(tests):
        gen_test.gradient_test(dense=dense, exp=exp, poprange=poprange, popsize=popsize, epochs=epochs)

        # Plots results of all algorithms
        currMaxes = np.array([node.stats["max"].max() for node in genSys.map])
        maxes += currMaxes
    
        speeds += np.array([node.stats["epoch"][node.stats["max"] == currMax].values[0] for node, currMax in zip(genSys.map, currMaxes)])

    plt.scatter(vars, speeds / tests, c=maxes / tests)

    plt.colorbar()
    plt.show()

def curve_metrics(genTest):
    for system in genTest.map:
        stats = system.stats
        plt.plot(stats["epoch"].values, stats["max"].values, label=system.id)
        
    plt.legend()
    plt.show()


# Creates genetic algorithm system with run and assess functions for the gradient
gen_test = GenSys([
    GradGen([
        gen.Group(size=2),
        gen.MeanSpreadCross(50, 0.1)
        ], id="circleSpread", assess="gradAssess"),

    GradGen([
        gen.Group(size=2),
        gen.OldMeanSpreadCross()
        ], id="weightSpread", assess="gradAssess")
    ], run="parallel")

#compMetrics(genSys, ["new", "old"], poprange=(100, 100, 100))
gen_test.gradient_test(dense=0.1, exp=0.8, poprange=(20,20,20), popsize=20, epochs=30)

curve_metrics(gen_test)
