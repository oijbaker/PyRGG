import rgg
import numpy as np

# create some random coordinates between 0 and 1
nodes = np.random.rand(50,2)

# create circular uniform points
thetas = np.random.rand(50)*2*np.pi
radii = np.random.rand(50)
nodes = np.array([radii*np.cos(thetas), radii*np.sin(thetas)]).T

sn = rgg.RandomGeometricGraph(nodes, r_0 = 0.3, domain_type="circle", domain_radius=1)
sn.plot_spectral_density(normalised=True)