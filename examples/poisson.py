from ngsolve import *
import ngs_petsc as petsc
from netgen.meshing import Mesh as NGMesh
from time import time

comm = mpi_world

if comm.rank==0:
    from netgen.geom2d import unit_square
    ngm = unit_square.GenerateMesh(maxh=0.01)
    ngm.Distribute(comm)
else:
    ngm = NGMesh.Receive(comm)
mesh = Mesh(ngm)

V = VectorH1(mesh, order=4, dirichlet='.*')
u,v = V.TnT()
a = BilinearForm(V)
a += SymbolicBFI(InnerProduct(grad(u),grad(v)))
f = LinearForm(V)
if v.dims[0]>1 or len(v.dims)>1:
    f += SymbolicLFI(v[0] + v[1])
else:
    f += SymbolicLFI(v)
f.Assemble()
gfu = GridFunction(V)
# c = Preconditioner(a, 'bddc')
a.Assemble()

petsc.Initialize()


opts = {"ksp_type":"cg", "ksp_atol":1e-30, "ksp_rtol":1e-8,
        #"pc_view" : "",
        #"ksp_view" : "",
        "ksp_monitor" : "",
        "pc_type" : "gamg"}
#        "pc_hypre_type" : "boomeramg"}
#opts = {"ksp_type" : "cg", "ksp_atol" : 1e-30, "ksp_rtol" : 1e-8}

mtdim = v.dims[0] if v.dims[0]>1 else v.dims[1]

gf_one = GridFunction(V)
kvecs = [gf_one.vec.CreateVector() for k in range(mtdim)]
if v.dims[0]>1 or len(v.dims)>1:
    gf_one.Set(CoefficientFunction((1,0)))
    kvecs[0].data = gf_one.vec
    gf_one.Set(CoefficientFunction((0,1)))
    kvecs[1].data = gf_one.vec
else:
    gf_one.Set(1)
    kvecs = [gf_one.vec.CreateVector()]
    kvecs[0].data = gf_one.vec
    
mat_wrap = petsc.PETScMatrix(a.mat, freedofs=V.FreeDofs(), format=petsc.PETScMatrix.AIJ)
mat_wrap.SetNearNullSpace(kvecs)
#mat_wrap = petsc.FlatPETScMatrix(a.mat, freedofs=V.FreeDofs())
ksp = petsc.KSP(mat=mat_wrap, name="someksp", petsc_options=opts, finalize=True)

# ngs_pc = petsc.NGs2PETScPrecond(mat=mat_wrap, pc=c)
#ksp.SetPC(ngs_pc)
#ksp.Finalize()

t = -time()
gfu.vec.data = ksp * f.vec
t += time()

quit()

mat_convert = petsc.PETScMatrix(a.mat, freedofs=V.FreeDofs())
# mat_convert.SetNearNullSpace(kvecs)
petsc_pc = petsc.PETSc2NGsPrecond(mat=mat_convert, name="reverse_someksp",
                                  petsc_options = {"pc_type" : "ml",
                                                   "pc_monitor" : "",
                                                   "pc_view" : ""})
t2 = -time()
solvers.CG(mat=a.mat, rhs=f.vec, sol=gfu.vec, pre=petsc_pc, tol=1e-8, printrates=comm.rank==0)
t2 += time()


ksp_res = ksp.results
if comm.rank==0:
    print('ndof ', V.ndofglobal)
    print(' pc used: ', ksp_res['pc_used'])
    print('ksp converged? ', ksp_res['conv_r'])
    print('PETSc took nits:', ksp_res['nits'])
    print('init. norm res: ', ksp_res['errs'][0])
    print(' fin. norm res: ', ksp_res['res_norm'])

if comm.rank==0:
    print('TSOL P', t)
    print('TSOL N', t2)

Draw(gfu, name='sol')

ex_sol = True
if ex_sol:
    err = f.vec.CreateVector()
    err.data = a.mat.Inverse(V.FreeDofs()) * f.vec - gfu.vec
    nr = Norm(err)
    if comm.rank==0:
        print('res ', nr)

petsc.Finalize()

