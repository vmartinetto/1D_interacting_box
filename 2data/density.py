import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

vecs = np.loadtxt('2part_Nx1000_L8_sc0.1_20200400_swss.dat')
x = np.linspace(0,8,1000)
densities = np.empty((1000,len(vecs[0,:])))
norm_densities = np.empty((1000,len(vecs[0,:]))) 

for i in range(len(vecs[0,:])):
    vec = vecs[:,i].reshape(1000,1000)
    dens = 2*np.transpose(vec).dot(vec)
    dens = np.diagonal(dens)
    densities[:,i] = dens

    norm = 2/simps(dens,x)
    norm_dens = norm*dens
    plt.plot(x,norm_dens)
    plt.show()
    plt.close()
    norm_densities[:,i] = norm_dens

np.savetxt('2part_Nx1000_L8_sc0.1_20200400_swss_dens.dat',densities)
np.savetxt('normed_2part_Nx1000_L8_sc0.1_20200400_swss_dens.dat',norm_densities)
