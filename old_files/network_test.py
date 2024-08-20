import omninet as net
import numpy as np
import matplotlib.pyplot as plt
import unittest



class TestSystem(unittest.TestCase):
    def setUp(self):
        self.System = net.System()

    def testNumberedId(self):
        """Tests node assignment with number ids"""

        # Finds current numberID in the map
        number = net.NodeID.numberID

        # Creates a node
        node = net.Node()

        # Assigns node and tests that node has correct map properties
        self.testAssign(node, number)

    def testStringId(self):
        """Tests node assignment with number ids"""

        # Creates node (with string id) and assigns it to the map
        node = net.Node(id="test")

        # Assigns node and tests that node has correct map properties
        self.testAssign(node, "test")

    def testPopulateMap(self):
        """Test populating the map with nodes"""
        # Populates map without specifying layers
        self.testPopulate(False)

        # Makes sure map matches input
        self.assertEqual(self.System.map, [
            [2],
            [3],
            [4, "stringId"]
        ])

    def testPopulateLayers(self):
        """Tests populating map at specific layers"""
        # Populates map at specific layers
        self.testPopulate([1,4,5])

        # Makes sure map matches input
        self.assertEqual(self.System.map, [
            [],
            [2],
            [],
            [],
            [3],
            [4, "stringId"]
        ])

    def testSysRun(self):
        """Test the map's propogation capability"""
        
        # Populates map with a variety of pointwise addition and subtraction nodes
        self.System.populate([
            [net.PointAdd([0,2]), net.PointSub([1,2]), net.PointSub([0,3])],
            [net.PointSub([0,1]), net.PointAdd([0,2])]
        ])

        # Runs netMap and confirms that output is correct
        result = np.stack(self.System.run(np.array([[0, 1, 2], [4, 3, 2], [10, 11, 12], [7, 7, 7]])))
        correct = np.array([[16, 20, 24], [3, 6, 9]])

        self.assertTrue(np.all(result == correct))

    def testDefaultRun(self):
        """Test that run function defaults to a pass-through if it returns None"""
        # Function that will return nothing
        def passFunct(input, way=True):
            pass

        # Creates node with run function
        node = net.Node(run=passFunct)

        # Tests that node passes value through
        self.assertEqual(node.run([0,1,2]), [0,1,2])

    def testSystemLogging(self):
        def recur(self, data, epochs):
            """Recursively runs system"""
            for epoch in range(epochs):
                # Runs system with data
                data = self.sequence(data)

                # Logs sum of all data
                sumData = np.sum(np.concatenate(data))

                self.log({"sum": [sumData]})
                

        testSys = net.System([
            [net.PointAdd("*"), net.PointSub("*")]
        ], run=recur)

        data = np.array([
            [1, 2, 1],
            [2, 1, 2]
        ])

        testSys.run(data, epochs=20)

        stats = testSys.stats

        plt.plot(stats["epoch"].values, stats["sum"].values, "ro")

        plt.show()


    
    # Specific testing functions
    def testPopulate(self, layers=None):
        """Test that nodes are correctly added to positions on map"""

        # If params are not defined, skip test
        if layers is None:
            self.skipTest("")
        elif layers == False:
            # If no layers are used, set layerNums to none
            layers = None
            

        # Adds nodes to map
        # Test individual nodes added to a layer and multiple nodes added to a layer
        # Tests adding Systems inside the System
        # Tests string ids as well as nodes
        self.System.populate([
            net.System(["stringId", net.Node()]),
            net.Node(),
            [net.Node(), "stringId"]
        ], layerNums=layers)

    def testAssign(self, node=None, id=None):
        """Tests that node is correctly registered"""

        # If parameters are none, skip the test
        if node is None: self.skipTest("")


        # Tests that node has correct id prop
        self.assertEqual(node.id, id)

        # Test that register gets node correctly
        self.assertIs(net.NodeID.get(id), node)

    def tearDown(self):
        # Clears node register
        net.NodeID.clear()


#if __name__ == "__main__":
    #unittest.main()

arr = [
    [1,2],
    [3,4],
    [5,6]
    ]

test = net.Tensor(arr, "x", "y")

print("x: %d" %(test.size("x")))
print("y: %d" %(test.size("y")))

print()

print(test.shape("x","y","y","x"))
print(test.shape())



