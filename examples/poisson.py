from ngsolve import *
import ngs_petsc as petsc
from netgen.meshing import Mesh as NGMesh
import ngs_amg

comm = mpi_world

# if comm.rank==0:
#     from netgen.geom2d import unit_square
#     ngm = unit_square.GenerateMesh(maxh=0.01)
#     ngm.Save('squarec.vol')
#     ngm.Distribute(comm)
# else:
#     ngm = NGMesh.Receive(comm)
# mesh = Mesh(ngm)

mesh = Mesh('squarec.vol', comm)
comm = MPI_Init()
V = H1(mesh, order=1, dirichlet='.*')
u,v = V.TnT()
a = BilinearForm(V)
a += SymbolicBFI(grad(u)*grad(v) + u * v)
f = LinearForm(V)
f += SymbolicLFI(v)
f.Assemble()
gfu = GridFunction(V)
c = Preconditioner(a, 'bddc')
a.Assemble()

#opts = {"ksp_type":"cg", "ksp_atol":1e-30, "ksp_rtol":1e-14, "pc_type":"none"}
opts = {"ksp_type":"cg", "ksp_atol":1e-30, "ksp_rtol":1e-14}
gfo = GridFunction(V)
gfo.Set(1)
gfo.vec[:] = 1
gfx = GridFunction(V)
gfx.Set(x)
gfy = GridFunction(V)
gfy.Set(y)

petsc.Initialize()

#ksp_res = petsc.KSPSolve(mat=a.mat, rhs=f.vec, sol=gfu.vec, kvecs=[gfo.vec], fds=V.FreeDofs(), **opts)

mat_wrap = petsc.FlatPETScMatrix(a.mat, freedofs=V.FreeDofs())
ksp = petsc.KSP(mat=mat_wrap, name="someksp", petsc_options=opts, finalize=False)

ngs_pc = ngs_amg.AMG_H1(blf=a, freedofs=V.FreeDofs(), energy="alg")
ngs_pc = petsc.NGs2PETSc_PC(mat=mat_wrap, pc=ngs_pc)
ksp.SetPC(ngs_pc)
ksp.Finalize()

gfu.vec.data = ksp * f.vec

ksp_res = ksp.results
if comm.rank==0:
    print('ndof ', V.ndofglobal)
    print(' pc used: ', ksp_res['pc_used'])
    print('ksp converged? ', ksp_res['conv_r'])
    print('PETSc took nits:', ksp_res['nits'])
    print('init. norm res: ', ksp_res['errs'][0])
    print(' fin. norm res: ', ksp_res['res_norm'])

Draw(gfu, name='sol')

err = f.vec.CreateVector()
err.data = a.mat.Inverse(V.FreeDofs()) * f.vec - gfu.vec
nr = Norm(err)
if comm.rank==0:
    print('res ', nr)

