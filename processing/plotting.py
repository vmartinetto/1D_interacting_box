import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

Nx = 17
x = np.linspace(0,1,Nx)
'''
ground = np.loadtxt('denspy-1001-9-2030.dat')
excited = np.loadtxt('denspy-1001-1-9-2030.dat')
third = np.loadtxt('denspy-1001-3-9-2030.dat')
plt.plot(ground,label='ground')
plt.plot(excited,label='first')
plt.plot(third,label='third')
plt.legend()
plt.savefig('charge_transfer_2030wells.png')
plt.show()
xc = np.loadtxt('vxc_1001_9-2030.dat')
ks = np.loadtxt('vks_1001_9-2030.dat')
plt.plot(x,xc,label='xc')
plt.plot(x,ks,label='ks')
plt.legend()
plt.savefig('charge_transfer_2030_vks.png')
plt.show()
plt.close()
vecs = np.loadtxt('evecs_1001_9-2030.dat')
for i in range(len(vecs[0,:])):
    plt.plot(x,vecs[:,i],label='evec_'+str(i))
plt.legend()
plt.savefig('ke_evecs_1001_9-2030.png')
plt.close()
vecs = np.loadtxt('vecs-1001-9-2030.dat')
for i in range(len(vecs[0,:])):
    plt.plot(x,vecs[:,i],label='evec_'+str(i))
plt.legend()
plt.show()
plt.savefig('evecs_1001_9-2030.png')
plt.close()

################no ionization implementation vs implementation xc#####################

xc = np.loadtxt('vxc_1001_9-2030.dat')
xc_ion = np.loadtxt('vxc_ION_1001_9-2030.dat')
plt.plot(x,xc,label='xc')
plt.plot(x,xc_ion,label='xc_ion')
plt.legend()
plt.show()
plt.close()
ks = np.loadtxt('vks_1001_9-2030.dat')
ks_ion = np.loadtxt('vks_ION_1001_9-2030.dat')
plt.plot(x,ks,label='ks')
plt.plot(x,ks_ion,label='ks_ion')
plt.legend()
plt.show()
plt.close()

#########################no ion imp vs imp ks evecs##################################

vecs = np.loadtxt('evecs_1001_9-2030.dat')
vecs_ion = np.loadtxt('evecs_ION_1001_9-2030.dat')
for i in range(len(vecs[0,:])):
    plt.plot(x,vecs[:,i],label='evec_'+str(i))
    plt.plot(x,vecs_ion[:,i],label='evec_ion_'+str(i))
    plt.legend()
    plt.show()
    plt.close()

################no ionization implementation vs implementation xc#####################

ks_ion = np.loadtxt('vks_ION_1001_9-2030.dat')
xc_ion = np.loadtxt('vxc_ION_1001_9-2030.dat')
plt.plot(x,ks_ion,label='ks_ion')
plt.plot(x,xc_ion,label='xc_ion')
plt.legend()
plt.savefig('charge_transfer_2030_vks_ion.png')
plt.show()
plt.close()

###################ionization evecs#################################################

vecs_ion = np.loadtxt('evecs_ION_1001_9-2030.dat')
for i in range(len(vecs_ion[0,:])):
    plt.plot(x,vecs_ion[:,i],label='evec_ion'+str(i))
plt.legend()
plt.savefig('ks_ION_evecs_1001_9-2030.png')
plt.show()
plt.close()

################################ionization ks density##########################33##

dens_ion = np.loadtxt('ks_dens_ION_1001-9-2030.dat')
for i in range(len(dens_ion[0,:])):
    plt.plot(x,dens_ion[:,i],label='dens_ion'+str(i))
plt.legend()
plt.savefig('ks_ION_dens_1001_9-2030.png')
plt.show()
plt.close()

###############################three-particle sc 1D a=.01##########################

vecs = np.loadtxt('kinectic_vectors_sc.01.dat')
ground = vecs[:,0].reshape(Nx,Nx,Nx)**2
ground2d = simps(ground)
plt.imshow(ground2d)
plt.show()
plt.close()
plt.plot(simps(ground2d))
plt.show()
plt.close()
print(simps(simps(ground2d)))

##############################three-particle sparse sc a=.01#######################

vecs = np.loadtxt('3part_Nx20_sc.01_sparse.dat')
Nx = 20
ground = vecs[:,0].reshape(Nx,Nx,Nx)**2
ground2d = simps(ground)
plt.imshow(ground2d)
plt.show()
plt.close()
plt.plot(simps(ground2d))
plt.show()
plt.close()
print(simps(simps(ground2d)))

vecs = np.loadtxt('3part_Nx21_sc.01_sparse.dat')
Nx = 21
ground = vecs[:,0].reshape(Nx,Nx,Nx)**2
ground2d = simps(ground)
plt.imshow(ground2d)
plt.show()
plt.close()
plt.plot(simps(ground2d))
#plt.savefig('3part_sc.01_Nx21.png')
plt.show()
plt.close()
print(simps(simps(ground2d)))
'''
vecs = np.loadtxt('3part_Nx21_L2_sc.01_sparse.dat')
Nx = 21
ground = vecs[:,0].reshape(Nx,Nx,Nx)**2
ground2d = simps(ground)
plt.imshow(ground2d)
plt.show()
plt.close()
plt.plot(simps(ground2d))
#plt.savefig('3part_sc.01_L2_Nx21.png')
plt.show()
plt.close()
print(simps(simps(ground2d)))

ground = vecs[:,1].reshape(Nx,Nx,Nx)**2
ground2d = simps(ground)
plt.imshow(ground2d)
plt.show()
plt.close()
plt.plot(np.diagonal(ground2d))
#plt.savefig('3part_sc.01_L2_Nx21.png')
plt.show()
plt.close()
print(simps(simps(ground2d)))

ground = vecs[:,2].reshape(Nx,Nx,Nx)**2
ground2d = simps(ground)
plt.imshow(ground2d)
plt.show()
plt.close()
plt.plot(simps(ground2d))
#plt.savefig('3part_sc.01_L2_Nx21.png')
plt.show()
plt.close()
print(simps(simps(ground2d)))
