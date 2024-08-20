# Author: Carson G Ray
# Language: Python
# Edition: P1-1


import numpy as np
import pandas as pd
from types import MethodType


class Node:
    def __init__(self, input="*", parent=None, id=None, run=None):
        """A data manipulation unit that can be part of a network in a System"""
        # Sets parent property
        self.parent = parent

        # Sets id property in register
        self.id = id
        self.id = NodeID.assign(self)

        # Sets node positions that the node receives input from in previous layer
        self.input = input
        
        # The function called when the node is run (set if parameter exists)
        if run:
            self.funct = MethodType(run, self)

        # Initiliazes test object to none
        self.__test = None

    def funct(self, input, **kwargs):
        """Default node operation function"""
        return input

    def run(self, input, **kwargs):
        """If run function does not return a value, pass the input through"""
        # If input has a length of 1, use only first value
        if len(input) == 1:
            input = input[0]
        
        # Gets result of function
        result = self.funct(input, **kwargs)

        # Returns input if function returns nothing
        return input if result is None else result    
        



      
class System(Node):
    def __init__(self, map=[], input=None, parent=None, id=None, run=None, assess=None, stats=None):
        """A network of nodes that each manipulate data"""
        # Calls Node initialization
        super().__init__(input, parent, id, run)

        # Defines map containing nodes
        self.map = []

        # Populates map with members
        self.populate(map)

        # Testing

        # Sets assessment function
        if assess:
            self.assess = MethodType(assess, self)

        # Initializes iterations to 0
        self.epoch = 0

        # Sets log statisitics (None uses all logged statistics, [] uses no statistics)
        self.statList = stats

        # Initilizes pandas database using list of statistics
        statList = [] if stats is None else stats
        
        self.stats = pd.DataFrame(columns=(["epoch"] + statList))

    def run(self, *args, **kwargs):
        # Clears log
        self.clear()

        # Sets epochs to 0
        self.epochs = 0

        # Calls run function with parameters
        result = self.funct(*args, **kwargs)

        return result if result else self

    def log(self, vals=None, newEpoch=True):
        """Logs stat variables"""
        # Creates dataframe to append
        append = pd.DataFrame(vals)
                
        # If stats are none, accept all variables
        statList = append.columns if self.statList is None else self.statList
                
        append = append[statList]

        # Add epoch column with length of append
        append["epoch"] = np.repeat(self.epoch, len(append))

        # Increments epoch if newEpoch is True
        if newEpoch:
            self.epoch += 1
                
        # Appends the new values and aligns columns
        self.stats = self.stats.append(append, ignore_index=True, sort=True)

    def clear(self):
        """Clears stat dataframe"""
        # Saves data
        result = self.stats
        
        # Resets to blank dataFrame
        self.stats = pd.DataFrame(columns=result.columns)

        # Returns data
        return result
    
    def populate(self, nodeLists, **kwargs):
        """Adds nodes to the system by layers"""
        # If nodeLists is empty, return
        if nodeLists == []:
            return

        
        # If layerNums does not exist, use range of nodelist
        layerNums = range(len(nodeLists)) if kwargs.get("layerNums") is None else kwargs["layerNums"]
            

        # Appends layers to map if they do not exist
        try:
            # Tries accessing deepest layer that will be appended to
            self.map[max(layerNums)]
        except IndexError:
            # Adds blank layers until the deepest layer is added
            self.map.extend([[] for i in range(max(layerNums) - len(self.map) + 1)])
                

        # Loops through elements of nodeLists and gets indices
        for layerNum, layer in zip(layerNums, nodeLists):
            # Encapsulates layer if not a list
            layer = encapList(layer)
                

            # Loops through nodes in layer and gets ids for each one assigns it to the system
            # if it does not have an id)
            for node in layer: 
                if isinstance(node, Node):
                    # If is a node, assign the parent property and get id
                    node.parent = self
                    id = node.id
                else:
                    # If it is not a node, just use the id
                    id = node
                    

                # Appends id to map
                self.map[layerNum].append(id)

    def run_pass(self, data, way=True):
        """Propogates data through map by layer"""
        # If the pass is backward, reverse map
        newMap = self.map if way else self.map[::-1]

        # Loops through layers of map (inidices)
        for layer in range(len(newMap)):
            # Sets data to result of running the layer
            data = self.layer(layer, data, way)
            

        return data

    # Default running function
    funct = run_pass

    # Default assess function
    assess = None

    def layer(self, layerNum, data, way=True):
        """Runs a single layer of the System"""
        # Gets current layer
        layer = self.map[layerNum]

        # Gets array of inputs for each node

        # Uses previous layer for inputs if pass is backwards
        try:
            inLayer = layer if way else self.map[layerNum + 1]
        except:
            # If no previous layer exists, use a direct input mapping
            inputs = np.arange(len(layer))
        else:
            # Initializes input list
            inputs = np.zeros(len(inLayer), dtype=object)

            # Loop through nodes in input layer to collect input mapping
            for i, id in enumerate(inLayer):
                # Gets node inputs from id
                nodeIn = NodeID.get(id).input
                
                # Handles special characters
                if nodeIn == "*":
                    # Includes all nodes in input
                    nodeIn = np.arange(len(inLayer))
                elif nodeIn == "^":
                    # Includes the node at current node's index in the input
                    nodeIn = i
                else:
                    nodeIn = np.array(nodeIn)
                
                # Sets input array in inputs list
                inputs[i] = nodeIn
        
        # Initilaizes newData
        newData = np.zeros(len(layer), dtype=object)

        # Loops through nodes in current layer to run them given their inputs
        for i, id in enumerate(layer):
            # Gets node
            node = NodeID.get(id)

            # Gets inputs for node
            if way:
                # If it is a forward pass, use direct input
                nodeIn = inputs[i]
            else:
                # If it is a backward pass, find inputs where the node index is referenced
                nodeIn = np.where(inputs == i)[0]

            # Returns result of running node function with inputs at specified indices
            newData[i] = node.run(data[nodeIn], way=way)
        
        # Returns new data (if it has a length of 1, use only item)
        if len(newData) == 1:
            return newData[0]
        else:
            return newData
        
                
                


# Specialized nodes
class PointAdd(Node):
    def funct(self, input, way):
        """Adds two inputs pointwise"""
        if way:
            return input[0] + input[1]
    
class PointSub(Node):
    def funct(self, input, way):
        """Subtracts two inputs pointwise"""
        if way:
            return input[0] - input[1]





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

def encapList(item):
    """If item is not a list encapsulate it in a list"""
    try:
        return list(item)
    except TypeError:
        return [item]

def scaleTo(array, min, max):
    """Scales array between min and max"""
    # Makes sure array is numpy array
    array = np.array(array)

    # Scales array
    return (array - array.min()) * (max - min) / (array.max() - array.min()) + min

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


        
