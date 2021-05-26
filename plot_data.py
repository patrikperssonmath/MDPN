import matplotlib.pyplot as plt
import numpy as np

f = open("./data_sparse/data.cvs", "r")

vals = np.loadtxt(f,delimiter=',')

f.close()

plt.plot(vals[0], vals[1])
plt.savefig('data.png')
