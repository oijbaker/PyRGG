import rgg
import numpy as np

mrgg = rgg.MarkovRandomGeometricGraph(
    connection_type="rayleigh", r_0=0.3, n=100, dimensions=3,
    rayleigh_beta=5, rayleigh_eta=2
)
mrgg.plot(wireframe=False)