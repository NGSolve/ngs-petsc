#Importing the needed library
from ngsolve import *
from netgen.geom2d import unit_square,SplineGeometry
import ngsolve.meshes as ngm
import netgen.gui
import sys
import ngs_petsc as ngp
from mpi4py import MPI
import petsc4py.PETSc as psc
import slepc4py.SLEPc as spc
import netgen.meshing
import numpy as np

comm = MPI.COMM_WORLD #I go and grab the comunicator

NEig = 10  #Number of eigenvalues to be computed
#Defining number of uniform mesh points
N=100
#Defining geometry
if comm.rank==0:
    geo = SplineGeometry()
    geo.AddRectangle((0,0),(np.pi,np.pi),bc="rect")
    mesh = Mesh(geo.GenerateMesh(maxh=1/N).Distribute(comm))
else:
    mesh = Mesh(netgen.meshing.Mesh.Receive(comm))
#Constructing the finite element space
# Velocity P2 Pressure P1
V = H1(mesh, order=1, dirichlet="rect")

#Init. test and trial functions
u = V.TrialFunction()
v = V.TestFunction()

#Defining the bilinear form needed to study the Eigenvalue problem

a = BilinearForm(V)
a += grad(u)*grad(v)*dx

m = BilinearForm(V)
m += u*v*dx 

a.Assemble() #Assembling stiffness matrix
m.Assemble() #Assembling mass matrix

#PETSC/SLEPC
#Exporting Stifness and Mass matrix to PETSc
A = ngp.PETScMatrix(a.mat, freedofs=V.FreeDofs()).GetPETScMat()
M = ngp.PETScMatrix(m.mat, freedofs=V.FreeDofs()).GetPETScMat()


gfu = GridFunction(V, multidim=NEig,name='eigs')

#Creating the Eigenvalue problem
E = spc.EPS().create()
E.setType(spc.EPS.Type.KRYLOVSCHUR)
E.setProblemType(spc.EPS.ProblemType.GNHEP);
E.setDimensions(NEig,spc.DECIDE);
E.setOperators(A,M)
ST = E.getST();
ST.setType(spc.ST.Type.SINVERT)
ST.setShift(1.0)
PC = ST.getKSP().getPC();
PC.setType("lu");
PC.setFactorSolverType("mumps");
E.setST(ST);
E.solve();
lam = [];

vmap = ngp.VecMap(V.ParallelDofs(), V.FreeDofs())
vmap(gfu.vec)
for s in range(NEig):
    xr = vmap.CreatePETScVector()
    xl = vmap.CreatePETScVector()
    lam.append(E.getEigenpair(s, xr, xl))
    gfu.vecs[s].data = vmap(xr);
if comm.rank==0:
    print(lam)
Draw(gfu) #Only in Sequential mode
