from old_core1 import *

import numpy as np
import matplotlib.pyplot as plt

class GenAlg(System):
    def generation(self, data):
        """Runs on generation of the genetic algorithm"""
        
        # If next generation, recursively map from results of last generation
        # Also clips population within ranges
        pop = data if self.epoch == 0 else self.clip(self.sequence(data))
        
        # Assesses current generation and adds to population
        data = self.add_pop(self.assess(pop.copy()))

        # Logs statistics
        self.log({
            "max": [data[1].max()],
            "min": [data[1].min()],
            "mean": [np.average(data[1])],
            "median": [np.median(data[1])],
            "valRange": [np.ptp(data[1])]
        })

        # Returns resulting population and values
        return data
        
    # Default run function
    run = generation

    def setup(self, ranges=None, size=None):
        """Creates random initial population"""
        # Sets popsize
        self.popsize = size
        
        # Sets default population range from 0 to 1
        poprange = [1]*size if ranges is None else ranges
        self.poprange = np.array(poprange)

        # Creates random population and stacks it on last axis to get array of points
        self.population = np.stack([np.random.random(size)*length for length in poprange], axis=-1)

        # Returns resulting population
        return self.population

    def add_pop(self, data):
        """Selects fittest individuals to continue population size"""
        # Unpacks both points and values
        points, values = data

        if self.epoch == 0:
            # If first epoch, reset population to current members
            self.population = points

            self.values = values
        else:
            # Appends individuals to current population and values
            self.population = np.concatenate((self.population, points))

            self.values = np.concatenate((self.values, values))

        # Sorts population by values and then sorts values
        self.population = self.population[np.argsort(self.values)[::-1]]

        self.values = np.sort(self.values)[::-1]

        # Trims population to population size
        self.population = self.population[0:self.popsize]

        self.values = self.values[0:self.popsize]

        # Returns population and values
        return self.population, self.values

    def clip(self, population):
        """Clips population within ranges"""
        # Swaps axes of population for clipping
        pop = population.copy().swapaxes(0,1)

        # Clips each axis of population within range
        pop = np.clip(pop, 0, self.poprange[:, np.newaxis])

        # Returns the swapped result
        return pop.swapaxes(0,1)




class Group(Node):
    def __init__(self, id=None, size=None, num=None):
        """Groups population into groups of a set size"""

        # Calls node initialization
        super().__init__(id)

        # Sets group size property
        self.size = size

        # Sets number of groups property
        self.num = num

    def run(self, data):
        # Unpacks both points and values
        points, values = data

        # Sets up points as list of points
        points = Tensor(points, "points", "coords")

        # Sets up values in same way
        values = Tensor(values, "values")

        # Number of points
        pointNum = points.size("points")

        if self.num is None:
            # Finds largest number of points that can be divided into sized groups
            newNum = pointNum - (pointNum % self.size)

            # Finds number of groups
            groups = int(newNum / self.size)
        else:
            # Finds number of points that can be divided into a number of groups
            newNum = pointNum - (pointNum % self.num)

            # Sets number of groups
            groups = int(newNum / self.num)
            
            
        # Trims population to new number of points
        points.vals = points.vals[:newNum]
        values.vals = values.vals[:newNum]

        size = int(newNum / groups)

        # Returns reshaped population into groups (adds axis to beginning
        points = points.on([1], "points", "coords").reshape(groups, size, points.size("coords"))
        values = values.on([1], "values").reshape(groups, size)

        return points, values





class MeanSpreadCross(Node):
    def __init__(self, id=None, expand=2, exp=0.14):
        # Calls node intialization
        super().__init__(id)

        # Sets expansion property
        self.expand = expand

        # Sets exponent property
        self.exp = exp
        
    def run(self, data):
        """Creates offspring by averaging characteristics of parent groups"""
        # Unpacks both points and values
        points, values = data

        # Sets up points as list of groups of points
        points = Tensor(points, "groups", "points", "coords")

        # Sets up values in same way but normalizes for each group
        values = Tensor(softmax(values, axis=-1), "groups", "values")

        # Gets valRange and scales
        scale = np.ptp(values.vals)*values.size("values")
        scale = self.expand * self.exp**scale + 1

        # Weights each point by the values and sums each group to get a weighted average
        average = np.sum(points.vals * values.on("groups", "values", [1]), axis=points.ax("points"))
        average = Tensor(average, "groups", "coords")

        # Finds the distance of all points to the weighted average and averages in each group
        dists = np.average(distance(points.vals, average.on("groups", [1], "coords")), axis=points.ax("points"))
        dists = dists * scale
        
        # Loops through each group and produces random points within a circle
        # extending from the weighted averages with a radius of the distances

        for center, radius in zip(average.vals, dists):
            # Creates random points within a circle
            randPoints = rand_circ(points.size("points"), center=center, radius=radius)

            # Create ouput points if array doesn't exist
            try:
                newPoints = np.append(newPoints, randPoints, axis=0)
            except NameError:
                newPoints = randPoints
        

        # Returns new points
        return newPoints
        




class OldMeanSpreadCross(Node):
    def __init__(self, id=None, variance=0.0001):
        """Creates offspring by averaging characteristics of parent groups"""
        
        # Calls node initialization
        super().__init__(id)

        # Sets variance property
        self.variance = variance

    def run(self, data):
        """Creates offspring by averaging characteristics of parent groups"""
        # Unpacks both points and values
        points, values = data

        # Sets up points as list of groups of points
        points = Tensor(points, "groups", "points", "coords")

        # Sets up values in same way
        values = Tensor(values, "groups", "values")

        # Creates array of weight adjustments and scales to particular variance
        # Each group has different weight adjustments to create a new offspring
        # group of the same size
        # Adjustment is array of matrices, with each matrix being one adjustment to the weights for all groups
        adjust = np.random.random(points.shape("points", "groups", "points"))*2*self.variance - self.variance

        adjust = Tensor(adjust, "adjusts", *values.axes)

        

        # Adds adjustments to values and normalizes values for each group
        values = softmax(np.squeeze(adjust.on("adjusts", "groups", "values") + values.on([1], "groups", "values")), axis=-1)
        values = Tensor(values, *adjust.axes)

        # Multiplies each point by the weights and sums each group.
        points = np.sum((values.on("adjusts", "groups", "values", [1]) * points.on([1], "groups", "points", "coords")), axis=2)
        points = Tensor(points, "adjusts", "groups", "coords")

        # Returns points and coords
        return points.vals.reshape(points.size("adjusts", "groups"), points.size("coords"))
        




        

# Utility functions
def point_grad(points, values, exp=0.9, size=None):
    """Creates a gradient between points in a dimensional space"""
    # Creates points tensor and converts values to np array
    points = Tensor(points, "coords", "points")
    values = np.array(values)
    
    # Creates list of ranges representing all possible coordinates in gradient
    ranges = [np.arange(length) for length in size]
    
    # Creates the cartesian product of all possible coordinates within size
    coords = Tensor(cartesian(ranges), "points", "coords")


    # Swaps axes of points so that it is a list of coordinates rather than a list of axes
    # Adds an axis to points so it is on the opposite orientation from the coords

    # Calculates distance between all possible coords and the selected gradient points
    dist = distance(coords.on("points", [1], "coords"), points.on([1], "points", "coords"))
    dist = Tensor(dist, "allPoints", "guidePoints")


    # The now two-dimensional array has the distance to each point in each column,
    # and the rows represent each possible point

    # The distances are decayed exponentially so that more distance has less weight
    # and negated so that more distance has less weight.
    # They are then weighted with the values array and averaged for each possible point
    newValues = np.average((exp**dist.on("allPoints", "guidePoints")) * values, axis=dist.ax("guidePoints"))

    # The result is a one-dimensional array of all of the possible points
    # with their corresponding values
    # This is reshaped back to the gradient space and returned
    return newValues.reshape(size)

def rand_grad(low, high, dense=0.01, exp=None, size=None):
    """Creates a random gradient by using a set of random points
    in the pointGrad function"""

    # Number of reference points is the total points in gradient times density
    pointNum = int(np.prod(size) * dense)

    # Creates random points by creating list of random coordinates for each axis
    # For each axis creates random intergers within the size of the axis
    points = np.stack([np.random.randint(length, size=pointNum) for length in size])

    # Creates random values for each point
    values = np.random.random(pointNum)*2 - 1

    # Creates gradient between the random points that is scaled to the end result
    return scaleTo(point_grad(points, values, exp=exp, size=size), low, high)

def rand_circ(num, center=[0, 0], radius=1):
    """Creates random points within a circle/sphere in multiple dimensions"""
    # Dimensions are equal to the length of the center point
    center = np.array(center)
    dims = len(center)
    
    # Creates array of random angles between 0 and 2pi
    angles = np.random.random((num, dims-1))*2*np.pi
    
    # Creates random magnitudes within radius
    mags = np.random.random((num, 1))*radius
    
    # Gets cosine of all angles and adds magnitudes to the beginning
    mags = np.append(mags, np.cos(angles), axis=-1)
    
    # Cumulatively sums cosines to get progressive magnitudes
    mags = np.cumprod(mags, axis=-1)
    
    # Multiplies pointwise by sines of angles and lets the last magnitude pass
    # through as the first coordinate
    points = mags * np.append(np.sin(angles), [[1]]*num, axis=-1)
    
    # Returns the points adjusted by the center
    return points + center

def distance(one, other):
    """Calculates vectorized distance between multidimensional points"""
    # Finds differences between arrays, squares them, and sums them and takes a square root
    # This completes the pythagorean theorem
    return np.sqrt(np.sum((one - other)**2, axis=-1))
        


if __name__ == "__main__":
    numPlots = 2

    f, axis = plt.subplots(1, numPlots)

    for i in range(numPlots):
        grad = rand_grad(-1, 1, dense=0.1, exp=0.9, size=(100,100))
        img = axis[i].imshow(grad, cmap=plt.get_cmap("terrain"))

    plt.colorbar(img, ax=axis)
    plt.show()
