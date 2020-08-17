from mpi4py import MPI
from ngsolve import *
import petsc4py.PETSc as psc
import ngs_petsc as ngp
import netgen.meshing

comm = MPI.COMM_WORLD

if comm.rank==0:
    from netgen.geom2d import unit_square
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.005).Distribute(comm))
else:
    mesh = Mesh(netgen.meshing.Mesh.Receive(comm))
    
V = H1(mesh, order=1, dirichlet='.*')
u,v = V.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(1*v*dx).Assemble()

gfu = GridFunction(V)

p4p_mat = ngp.PETScMatrix(a.mat, freedofs=V.FreeDofs()).GetPETScMat()

ksp = psc.KSP().create()
ksp.setOperators(p4p_mat)
ksp.setType(psc.KSP.Type.CG)
ksp.setNormType(psc.KSP.NormType.NORM_NATURAL)
ksp.getPC().setType("gamg")
ksp.setTolerances(rtol=1e-6, atol=0, divtol=1e16, max_it=50)

ksp.view()

if comm.rank==0:
    ksp.setMonitor(lambda a, b, c: print("it", b, "err", c))

# can map between NGSolve and PETSc vectors
vmap = ngp.VecMap(V.ParallelDofs(), V.FreeDofs())

psc_rhs = vmap(f.vec)
psc_sol = vmap.CreatePETScVector()
ksp.solve(b=psc_rhs, x=psc_sol)
gfu.vec.data = vmap(psc_sol)


# to compare sequential and parallel version:
ip = InnerProduct(f.vec, gfu.vec)

if comm.rank==0:
    print('ndof:', V.ndofglobal)
    print('pc used:', ksp.getPC().getType())
    print('took nits:', ksp.getIterationNumber())
    print('inner product:', ip)


# only available in sequential mode
Draw(gfu, name='sol')
