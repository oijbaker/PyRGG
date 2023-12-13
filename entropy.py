import rgg
import numpy as np
import matplotlib.pyplot as plt

# create some random coordinates between 0 and 1
nodes = np.random.rand(100,2)

# create circular uniform points
# thetas = np.random.rand(50)*2*np.pi
# radii = np.sqrt(np.random.rand(50))
# nodes = np.array([radii*np.cos(thetas), radii*np.sin(thetas)]).T

vn_entropy = []
r2_entropy = []
r3_entropy = []
r100_entropy = []
r1_1_entropy = []

for r0 in np.arange(0, 1, 0.01):
    sn = rgg.RandomGeometricGraph(nodes, r_0 = r0, domain_type="rectangle")
    vn_entropy.append(sn.get_von_neumann_entropy())
    # r2_entropy.append(sn.get_renyi_entropy(2))
    # r3_entropy.append(sn.get_renyi_entropy(3))
    # r100_entropy.append(sn.get_renyi_entropy(100))
    # r1_1_entropy.append(sn.get_renyi_entropy(1.1))

plt.plot(np.arange(0, 1, 0.01), vn_entropy, label="von Neumann entropy", color="red", linestyle="--")
# plt.plot(np.arange(0, 1, 0.01), r1_1_entropy, label="Renyi entropy (alpha=1.1)", color="orange", linestyle="--")
# plt.plot(np.arange(0, 1, 0.01), r2_entropy, label="Renyi entropy (alpha=2)", color="blue", linestyle="--")
# plt.plot(np.arange(0, 1, 0.01), r3_entropy, label="Renyi entropy (alpha=3)", color="green", linestyle="--")
# plt.plot(np.arange(0, 1, 0.01), r100_entropy, label="Renyi entropy (alpha=100)", color="purple", linestyle="--")
plt.legend()
plt.show()

gradients = []
for i in range(len(vn_entropy)-1):
    gradients.append(vn_entropy[i+1]-vn_entropy[i])
plt.plot(np.arange(0, 1, 0.01)[:-1], gradients, label="Gradient of von Neumann entropy", color="red", linestyle="--")
plt.show()