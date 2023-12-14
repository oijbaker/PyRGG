import rgg
import numpy as np
import matplotlib.pyplot as plt

# create some random coordinates between 0 and 1
nodes = np.random.rand(400,2)

nodes = nodes*20

sn = rgg.RandomGeometricGraph(nodes, r_0 = 0.1, domain_type="rectangle",
                              connection_type="waxman", 
                              waxman_a=1, waxman_beta=2,
                              domain_height=20, domain_width=20)

sn.plot()