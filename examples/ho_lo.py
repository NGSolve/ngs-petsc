from ngsolve import *
import ngs_petsc as petsc
from netgen.meshing import Mesh as NGMesh
from time import time

comm = mpi_world

petsc.Initialize()

if comm.rank==0:
    from netgen.csg import unit_cube
    ngm = unit_cube.GenerateMesh(maxh=0.1)
    ngm.Distribute(comm)
else:
    ngm = NGMesh.Receive(comm)
ngm.Refine()
mesh = Mesh(ngm)

V = H1(mesh, order=5, dirichlet='.*', wb_withoutedges=True)
u,v = V.TnT()
a = BilinearForm(V)
a += SymbolicBFI(grad(u)*grad(v))
if True:
    c = Preconditioner(a, "bddc", coarsetype="petsc_pc", petsc_pc_petsc_options = ["pc_type ksp",
                                                                               # "ksp_ksp_monitor",
                                                                               # "ksp_ksp_view_converged_reason",
                                                                               # "help"
                                                                               "ksp_pc_type gamg"])
else:
    c = Preconditioner(a, "petsc_pc", petsc_pc_petsc_options = ["pc_type gamg"])
a.Assemble()

f = LinearForm(V)
f += SymbolicLFI(v)
f.Assemble()

gfu = GridFunction(V)


t2 = -time()
solvers.CG(mat=a.mat, rhs=f.vec, sol=gfu.vec, pre=c, tol=1e-8, printrates=comm.rank==0)
t2 += time()

if comm.rank == 0:
    print('---------------------------')
    print('ndof = ', V.ndofglobal)
    print('lo ndof = ', V.lospace.ndofglobal)
    print('time solve = ', t2)
    print('dofs / (sec * core) = ', V.ndofglobal / (t2 * max(comm.size-1, 1)))
    print('---------------------------')

from ngsolve.la import EigenValues_Preconditioner
ngsglobals.msg_level = 0
evs = EigenValues_Preconditioner(a.mat, c)
if comm.rank == 0:
    print('min, max ev = ', evs[0], '/', evs[len(evs)-1], ',   Condition = ', evs[len(evs)-1]/evs[0])

Draw(gfu, name='sol')

ex_sol = False
if ex_sol:
    err = f.vec.CreateVector()
    err.data = a.mat.Inverse(V.FreeDofs()) * f.vec - gfu.vec
    nr = Norm(err)
    if comm.rank==0:
        print('res ', nr)

