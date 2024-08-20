# Author: Carson G Ray
# Language: Python
# Edition: P1-1


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Node:
    def __init__(self, id=None, run=None):
        """A data manipulation unit that can be part of a network in a System"""
        # Sets id property in register
        self.id = id
        self.id = NodeID.assign(self)
        
        # The function called when the node is run (set if parameter exists)
        if run:
            self.run = getattr(self, run)
        



      
class System(Node):
    def __init__(self, map=[], id=None, run=None, assess=None):
        """A network of nodes that each manipulate data"""
        # Calls Node initialization
        super().__init__(id, run)

        # Defines map containing nodes
        self.map = []

        # Populates map with members
        self.populate(map)

        # Testing

        # Sets assessment function
        if assess:
            self.assess = getattr(self, assess)

        # Initializes iterations to 0
        self.epoch = 0

        # Initilizes pandas dataframe with epochs column
        self.stats = pd.DataFrame(columns=["epoch"])

    def log(self, vals):
        """Logs stat variables"""
        # Creates dataframe to append
        append = pd.DataFrame(vals)

        # Add epoch column with length of append
        append["epoch"] = np.repeat(self.epoch, len(append))
                
        # Appends new values by epoch
        self.stats = self.stats.append(append, ignore_index=True, sort=True)

    def clear(self):
        """Clears stat dataframe"""
        # Saves data
        result = self.stats
        
        # Resets to blank dataFrame
        self.stats = pd.DataFrame(columns=result.columns)

        # Returns data
        return result
    
    def populate(self, nodeList):
        """Adds nodes to the system"""
        # Loops through each item in the node list
        for node in nodeList: 
            if isinstance(node, Node):
                # If is a node, assign the parent property
                node.parent = self
                node = node
            else:
                # If it is id, get node from id
                node = NodeID.get(node)
                        
            # Appends node to map
            self.map.append(node)

    def sequence(self, data, way=True):
        """Propogates data through map by layer"""
        # If the pass is backward, reverse map
        newMap = self.map if way else self.map[::-1]

        # Loops through layers of map
        for node in newMap:
            # Sets direction of node if availiable
            if hasattr(node, "way"):
                node.way = way
            # Sets data to result of running the layer
            data = node.run(data)
            

        return data

    # Default running function
    run = sequence

    # Default assess function
    assess = None

    def parallel(self, data):
        """Propogates data through map running each layer in parallel"""
        # Splits data if it does not start with multiple tracks
        if len(data) == 1:
            data = data*len(self.map)
        
        # Loops through each layer
        newData = []
        
        for node, thread in zip(self.map, data):
            # Sets epoch of node equal to parent epoch
            node.epoch = self.epoch
            
            # Runs each node on data thread
            newData.append(node.run(thread))

        # Returns tuple of data threads
        return tuple(newData)

    def recur(self, data, epochs):
        """Recursively runs system and passes data to next iteration"""
        # Clears log and sets epochs to 0
        self.clear()
        self.epoch = 0
        
        # Loops through epochs and runs function
        for epoch in range(epochs):
            data = self.run(data)
            self.epoch += 1

        # Returns result
        return data





class Register:
    def __init__(self):
        # Node registry
        self.nodes = {}

        # Amount of numbered nodes
        self.numberID = 0

    def assign(self, node):
        """Assigns nodes to global class and returns id"""
        if node.id is None:
            # If id is not set, use number id
            nodeID = self.numberID

            # Increment number id
            self.numberID += 1
        else:
            nodeID = node.id

        # Assigns node to registry
        self.nodes[nodeID] = node

        # Returns id
        return nodeID

    def get(self, id):
        # Gets node from id
        return self.nodes[id]

    def clear(self):
        # Clears register
        self.nodes = {}
        self.numberID = 0
    

# Creates new register
NodeID = Register()



class Tensor:
    def __init__(self, array, *axes):
        """A multi-dimensional array construct with labeled axes"""
        # Defines tensor values from numpy array
        self.vals = np.array(array)

        # Defines axes of tensor
        axes = list(axes)
        self.axes = axes

        # Defines shape of vals
        shape = self.vals.shape

        if len(axes) > len(shape):
            # Fills out shape if there are more axes than current shape
            self.vals = self.on([len(axes) - len(shape)])
        else:
            # Appends axis numbers to axes if not all axes are defined
            self.axes = axes + list(range(len(axes), len(shape)))
        
    def ax(self, *labels):
        """Determines axis number of label(s)"""
        axes = []
        
        for label in labels:
            try:
                # Tries finding string label
                axes.append(self.axes.index(label))
            except ValueError:
                # Uses actual index of axis
                axes.append(int(label))

        # If axes has length of one, return first item
        if len(axes) == 1:
            return axes[0]
        else:
            return axes
    
    def on(self, *args):
        """Selects axis of a tensor to return a numpy array"""
        
        # Sets result array to a copy of tensor values
        result = self.vals.copy()
        
        # Sets num to count axis number
        num = 0

        # New axes start at the end of the result shape
        newStart = len(result.shape)

        # List to hold source of new axes
        axSource = np.array([], dtype=int)

        # List to hold destination of new axes
        axDest = np.array([], dtype=int)
        
        # Loops through arguments
        for arg in args:
            if isinstance(arg, list):
                # If argument is newaxis arg specified by
                # [num] where num is number of new axes
                
                # Gets number of new axes
                new = arg[0]
                
                # Creates full slices along length of axes
                slices = [np.index_exp[:][0]]*(len(result.shape))
                
                # Adds axes to array plus the amount of new axes
                result = result[tuple(slices + [np.newaxis]*new)]

                # Adds the new axes to the axis order
                axSource = np.append(axSource, np.arange(newStart, newStart + new))

                # Adds current axis numbers to the axis destination
                axDest = np.append(axDest, np.arange(num, num + new))
                
                # Increments num and newCount by number of new axes
                num += new

                newStart += new
                
            elif arg is Ellipsis:
                # If argument is ellipsis, assume that all axis referencing is complete
                # Sets axis num to the current shape
                num = len(result.shape)
                
            else:
                # Argument is axis label
                
                # Swaps axis specified by label with current position
                axSource = np.append(axSource, [self.ax(arg)])

                # Adds current axis num
                axDest = np.append(axDest, [num])
                
                # Increments num
                num += 1
            
        # Transposes result with new axOrder mapped to the sequence of new axes
        return np.moveaxis(result, source=axSource, destination=axDest)

    def shape(self, *axes):
        """Returns the shape of the tensor along axes"""
        if len(axes) == 0:
            # If no parameters are passed, return full shape of tensor
            return self.vals.shape
        else:
            # Gets shape from each axis
            return tuple([self.vals.shape[self.ax(ax)] for ax in axes])

    def size(self, *axes):
        """Returns the size of the tensor along axes"""
        return np.prod(self.shape(*axes))
            




# Utility functions

def scaleTo(array, min, max):
    """Scales array between min and max"""
    # Makes sure array is numpy array
    array = np.array(array)

    # Scales array
    return (array - array.min()) * (max - min) / np.ptp(array) + min

def cartesian(arrays):
    """Returns the cartesian product of the arrays"""

    # Creates a meshgrid of each array and then flattens them
    axes = [axis.flatten() for axis in np.meshgrid(*arrays)]

    # Stacks the arrays along the last axis to create a list
    # of all possible combinations
    return np.stack(axes, axis=-1)

def softmax(array, axis=-1):
    """Runs softmax normalization along axis"""
    # Exponentializes array by e
    exp = np.exp(array)
    
    # Gets sum of exponentializes array along an axis
    # and then replaces the axis for broadcasting
    expSum = np.moveaxis(np.sum(exp, axis=axis)[np.newaxis, ...], 0, axis)
    
    # Returns exp array normalized by sum
    return (exp / expSum)
        
        


        
