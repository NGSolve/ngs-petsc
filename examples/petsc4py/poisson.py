from mpi4py import MPI
from ngsolve import *
import petsc4py as psc
import ngs_petsc as ngp
# import sys

comm = MPI.COMM_WORLD

from netgen.geom2d import unit_square
if comm.rank==0:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.05).Distribute(comm))
else:
    mesh = Mesh(netgen.meshing.Mesh.Receive(comm))

    
V = H1(mesh, order=1, dirichlet='.*')
u,v = V.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(1*v*dx).Assemble()

gfu = GridFunction(V)

# ngp.Initialize()

wrapped_mat = ngp.PETScMatrix(a.mat, freedofs=V.FreeDofs()) #  format=ngp.PETScMatrix.IS_AIJ)

# gives access to the petsc4py mat
p4p_mat = wrapped_mat.GetPETScMat()
# if comm.size > 1:
#    p4p_mat.convert("mpiaij") 

# this is a wrapper around a KSP
wrapped_ksp = ngp.KSP(mat=wrapped_mat, name="someksp", petsc_options={"ksp_type":"cg", "pc_type" : "gamg"}, finalize=False)

# this is the petsc4py KSP
ksp = wrapped_ksp.GetKSP()
ksp.setTolerances(1e-6, 0, 1e12, 20)
if comm.rank==0:
    ksp.setMonitor(lambda a, b, c: print("it", b, "err", c))

wrapped_ksp.Finalize()

# can map between NGSolve and PETSc vectors
vec_map = wrapped_mat.GetRowMap()
psc_rhs = vec_map.CreatePETScVector()
psc_sol = vec_map.CreatePETScVector()


t1 = Timer("solve_psc")
t1.Start()
vec_map.NGs2PETSc(f.vec, psc_rhs)
ksp.solve(b=psc_rhs, x=psc_sol)
vec_map.PETSc2NGs(gfu.vec, psc_sol)
t1.Stop()

if comm.rank==0:
    print('ndof ', V.ndofglobal)
    print('tsol', t1.time)
    print(' pc used: ', ksp.getPC().getType())
    print('took nits:', ksp.getIterationNumber())

Draw(gfu, name='sol')

