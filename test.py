import spatialnetwork
import numpy as np

# create some random coordinates between 0 and 1
nodes = np.random.rand(50,2)
sn = spatialnetwork.SpatialNetwork(nodes, r_0 = 0.2)
sn.plot()