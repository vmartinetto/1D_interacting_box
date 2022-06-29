import numpy as np
from scipy.integrate import simps
import scipy.sparse as spa
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import math
from scipy.sparse.linalg import eigsh

###################################Kronecker Delta#################################

def kron(a,b):
    '''
    A simple Kronecker delta implementation
    Input
        a,b: Real, Float, Int
            Any real, float, or integer value
    Output
        1 or 0:
            returns 1 if the value are the same, returns 0 if they do not
    '''
    if a==b:
        return 1
    else:
        return 0

#############################Kinectic Element Calculator############################ 

'''

Formation of the dense Kinetic Energy Matrix using the function Kin

mat = np.empty((Nx**2,Nx**2))
for p in range(Nx):
    for q in range(Nx):
        for i in range(Nx):
            for j in range(Nx):
                n = (p*Nx)+q
                m = (i*Nx)+j
                mat[n,m] = Kin(p,q,i,j)
mat = mat/dx**2
np.fill_diagonal(mat,np.diagonal(mat)+np.tile(vext,Nx))

'''

def Kin(p,q,i,j):
    '''
    Calculates the value of an element of the 1D three particle Kinetic
    Energy matrix given the six values p,q,r,i,j,k.

    INPUT
        p,q,r,i,j,k: Int
            The indices of the left and right eigenfunctions given three
            particles confined to a one dimensional box. 
    OUTPUT
        Knm: real, float
            The value of the kinetic energy matrix at indixes n and m given 
            the single particle basis functions are delta functions.
    '''
    qj = kron(q,j)*(kron(p,i-1)-2*kron(p,i)+kron(p,i+1))
    pi = kron(p,i)*(kron(q,j-1)-2*kron(q,j)+kron(q,j+1))
    Knm = -(1/2)*(qj+pi)
    return Knm

###########################Soft-Coulomb Interaction Calculator#####################

def Int(Nx,dx,a):
    '''
    Calculates the vector that lies along the main diagonal of the interaction potetnial
    matrix given delta function basis fnuctions in a 1D box. This is a soft-coulomb interaction
    not a full coulomb. A full coulomb would require special processing to solve in 1D.

    INPUT
        Nx: Int
            The number of wanted gridpoints in the 1D box.
        dx: float
            The grid spacing in the 1D box. dx*(Nx-1) = Length of box
        a: float
            The softening parameter of the soft-coulomb interaction.
    OUTPUT
        vint: np.array, vector, len=Nx**3
            The soft-coulomb interaction that lies along the main diagonal of the 
            interaction matrix given delta function basis functions in a 1D box.
    '''
    vint = np.empty(Nx**2)
    for i in range(Nx):
        for j in range(Nx):
            m = (i*Nx)+j
            vint[m]= (1 / math.sqrt(dx ** 2 * (i-j) ** 2 + a ** 2)) 
    return vint

#############################3-particle Sparse Operators##########################

def Int_sparse(Nx,dx,a):
    '''
    Mostly a wrapper around Int. makes the sparse matrix from Int and returns it.

    INPUT
        Nx: Int
            The number of wanted gridpoints in the 1D box.
        dx: float
            The grid spacing in the 1D box. dx*(Nx-1) = Length of box
        a: float
            The softening parameter of the soft-coulomb interaction.
    OUTPUT
        W: scipy.sparse.dia_matrix, shape=(Nx**2,Nx**2)
            A scipy sparse matrix object with vint from Int along the main diagonal.
    '''
    vint = Int(Nx,dx,a)
    W = spa.dia_matrix((vint,0),shape=(Nx**2,Nx**2))
    return W

def Sparse_Kin_2par(Nx,dx,vext):
    '''
    Constructs the kinetic energy matrix for three interacting particles in a 1D
    box given a delta function basis set. Using a three point centeral finite differnce for
    the second derivative.

    INPUT
        Nx: Int
            The number of wanted gridpoints in the 1D box.
        dx: float
            The grid spacing in the 1D box. dx*(Nx-1) = Length of box
        vext: np,array, vector, len=Nx
            A vector containg the external potential within the 1D box. It 
            is repeated Nx**2 times over the main diagonal of K.
    OUTPUT
        K: scipy.sparse.dia_matrix, shape=(Nx**3,Nx**3) 
            A scipy sparse matrix object with the bands of the kinetic matrix as well
            as the external potnetial.

    '''

    vext1 = spa.dia_matrix((vext,[0]),shape=(Nx,Nx))
    vext2 = spa.kron(vext1,spa.identity(Nx)) + spa.kron(spa.identity(Nx),vext1)

    # make the diagonals of the sparse matrix
    main = np.ones(Nx**2)*(2/dx**2)
    off1 = np.ones(Nx**2-1)*(-.5/dx**2)
    offNx = np.ones(Nx**2-Nx)*(-.5/dx**2)

    #add zeroes where necessary
    for i in range(Nx-1):
        off1[(Nx*(i+1))-1] = 0

    #pad vectors
    offu1 =  np.append([0],off1)
    offd1 =  np.append(off1,[0])
    offuNx = np.append(np.zeros(Nx), offNx)
    offdNx = np.append(offNx, np.zeros(Nx))

    #construct the diagonal matrix
    #diags = np.array([main+np.tile(vext,Nx), offd1, offu1, offdNx, offuNx])
    diags = np.array([main, offd1, offu1, offdNx, offuNx])
    #print(diags)
    K = (spa.dia_matrix((diags, [0, -1, 1, -Nx, Nx]), 
        shape= (Nx**2,Nx**2))
            )

    return K + vext2

###################################Matrix Visulization###########################

def Mat_view(mat):
    plt.spy(mat)
    plt.title('Mat View')
    plt.show()
    plt.close()
    return 0

def Sparse_mat_view(mat):

    plt.spy(mat.todense())
    plt.title('Sparse Mat View')
    plt.show()
    plt.close()
    return 0

########################################MAIN####################################

if __name__ == '__main__':
    Nx = 1000
    L = 8
    a = .1
    x = np.linspace(0,L,Nx)
    dx = np.abs(x[1]-x[0])

    vext = np.zeros(len(x))

    for i in range(Nx):
        if (dx*i > 1) and (dx*i < 2):
            vext[i] = 20
        if (dx*i > 4) and (dx*i < 5):
            vext[i] = 200
        if (dx*i > 6) and (dx*i < 7):
            vext[i] = 400


    K = Sparse_Kin_2par(Nx,dx,vext)
    V = Int_sparse(Nx,dx,a)
    ham = K+V
    vals, vecs = eigsh(ham, which='SA')
    print(vals*27.2114)
    np.savetxt('2data/2part_Nx'+str(Nx)+'_L'+str(L)+'_sc'+str(a)+'_20200400_swss.dat', vecs, fmt='%.9e', delimiter=' ')
    np.savetxt('2data/2part_Nx'+str(Nx)+'_L'+str(L)+'_sc'+str(a)+'_20200400_swss_vals.dat', vals, fmt='%.9e', delimiter=' ')
