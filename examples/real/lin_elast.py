from ngsolve import *
import ngs_petsc as petsc
import netgen.geom2d as geom2d
from netgen.meshing import Mesh as NGMesh

comm = mpi_world

dim, geo = 2, geom2d.SplineGeometry()
geo.AddRectangle((0, 0), (10,1), leftdomain=1, rightdomain=0, bcs = ("left", "outer", "outer", "outer"))
geo.SetMaterial(1, "mat")

if comm.rank==0:
    ngm = geo.GenerateMesh(maxh=0.05)
    ngm.Distribute(comm)
else:
    ngm = NGMesh.Receive(comm)
    ngm.SetGeometry(geo)
mesh = Mesh(ngm)


multidim = dim
V = H1(mesh, order=1, dirichlet="left", dim = multidim)
trials, tests = V.TnT()
u = CoefficientFunction(tuple(trials[k] for k in range(dim)))
gradu = CoefficientFunction( tuple(grad(trials)[i,j] for i in range(dim) for j in range(dim)), dims=[dim,dim])
epsu = 0.5 * (gradu + gradu.trans)
ut = CoefficientFunction(tuple(tests[k] for k in range(dim)))
gradut = CoefficientFunction( tuple(grad(tests)[i,j] for i in range(dim) for j in range(dim)), dims=[dim,dim])
epsut = 0.5 * (gradut + gradut.trans)
a = BilinearForm(V, symmetric=False)
a += SymbolicBFI( InnerProduct(epsu, epsut) )
force = CoefficientFunction( (0, 0.0001, 0) if dim==3 else (0, 0.002)) 
# force = CoefficientFunction( (0.0001, 0, 0) if dim==3 else (0.002, 0)) 
f = LinearForm(V)
f += SymbolicLFI( force*ut )
f.Assemble()
gfu = GridFunction(V)

a.Assemble()

def rb_modes(fes):
    dim = fes.mesh.dim
    if dim == 3:
        RBMS = [(1,0,0), (0,1,0), (0,0,1),
                (y,-x,0), (0,z,-y), (-z, 0, x)]
    else:
        RBMS = [(1,0), (0,1), (y,-x)]
    gfu = GridFunction(fes)
    rbm_vecs = list()
    upart = gfu
    for RBM in RBMS:
        if dim==3:
            ucf = CoefficientFunction((RBM[0], RBM[1], RBM[2]))
        else:
            ucf = CoefficientFunction((RBM[0], RBM[1]))
        upart.Set(ucf)
        v = gfu.vec.CreateVector()
        v.data = gfu.vec
        rbm_vecs.append(v)
    return rbm_vecs

petsc.Initialize()

mat_wrap = petsc.PETScMatrix(a.mat, freedofs=V.FreeDofs(), format=petsc.PETScMatrix.AIJ)
opts = {"ksp_type":"cg", "ksp_atol":1e-30, "ksp_rtol":1e-8, "pc_type":"ml"}
ksp = petsc.KSP(mat=mat_wrap, name="someksp", petsc_options=opts, finalize=False)
ksp.GetMatrix().SetNearNullSpace(rb_modes(V))

# import ngs_amg
# mat_wrap = petsc.FlatPETScMatrix(a.mat, freedofs=V.FreeDofs())
# ngs_amg_opts = {"energy" : "alg", "comp_sm" : True, "force_comp_sm" : True, "max_cv" : 500, "ass_frac" : 0.15, "skip_ass" : 3}
# ngs_pc = ngs_amg.AMG_EL2(blf=a, freedofs=V.FreeDofs(), **ngs_amg_opts)
# ngs_pc = petsc.NGs2PETSc_PC(mat=mat_wrap, pc=ngs_pc)
# ksp.SetPC(ngs_pc)

ksp.Finalize()

from time import time
t = -time()
gfu.vec.data = ksp * f.vec
t += time()

if comm.rank==0:
    print('TSOL P', t)

ksp_res = ksp.results
if comm.rank==0:
    print('ndof ', V.ndofglobal)
    print(' pc used: ', ksp_res['pc_used'])
    print('ksp converged? ', ksp_res['conv_r'])
    print('PETSc took nits:', ksp_res['nits'])
    print('init. norm res: ', ksp_res['errs'][0])
    print(' fin. norm res: ', ksp_res['res_norm'])

petsc.Finalize()

