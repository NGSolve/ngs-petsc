from ngsolve import *
import ngs_petsc as petsc
from netgen.meshing import Mesh as NGMesh

comm = mpi_world

# if comm.rank==0:
#     from netgen.geom2d import unit_square
#     ngm = unit_square.GenerateMesh(maxh=0.01)
#     ngm.Save('squarec.vol')
#     ngm.Distribute(comm)
# else:
#     ngm = NGMesh.Receive(comm)
# mesh = Mesh(ngm)

mesh = Mesh('square.vol', comm)
comm = MPI_Init()
V = H1(mesh, order=1, dirichlet='.*')
u,v = V.TnT()
a = BilinearForm(V)
a += SymbolicBFI(grad(u)*grad(v) + u * v)
f = LinearForm(V)
f += SymbolicLFI(v)
f.Assemble()
gfu = GridFunction(V)
#c = Preconditioner(a, 'bddc')
a.Assemble()

gfo = GridFunction(V)
gfo.Set(1)
gfo.vec[:] = 1
gfx = GridFunction(V)
gfx.Set(x)
gfy = GridFunction(V)
gfy.Set(y)

petsc.Initialize()

#ksp_res = petsc.KSPSolve(mat=a.mat, rhs=f.vec, sol=gfu.vec, kvecs=[gfo.vec], fds=V.FreeDofs(), **opts)


ex_sol = False

#opts = {"ksp_type":"cg", "ksp_atol":1e-30, "ksp_rtol":1e-8, "pc_type":"ml", "pc_ml_PrintLevel" : "3"}

opts = {"ksp_type":"cg", "ksp_atol":1e-30, "ksp_rtol":1e-8, "pc_type":"ml", "pc_ml_PrintLevel" : "3",
        "pc_mg_cycles" : "1", "pc_ml_maxNlevels" : "10", "pc_mg_log" : ""}

#opts = {"ksp_type":"cg", "ksp_atol":1e-30, "ksp_rtol":1e-8}

opts["ksp_plot_eigenvalues"] = ""
opts["ksp_compute_eigenvalues"] = ""
opts["ksp_monitor"] = ""
opts["ksp_view"] = ""
opts["pc_mg_view"] = ""
opts["pc_mg_log"] = ""

mat_wrap = petsc.PETScMatrix(a.mat, freedofs=V.FreeDofs())
#mat_wrap = petsc.FlatPETScMatrix(a.mat, freedofs=V.FreeDofs())
ksp = petsc.KSP(mat=mat_wrap, name="someksp", petsc_options=opts, finalize=False)

import ngs_amg
ngs_amg_opts = {"energy" : "alg", "comp_sm" : True, "force_comp_sm" : True, "max_cv" : 10, "ass_frac" : 0.15, "skip_ass" : 3,
                "prol_max_per_row" : 10, "prol_min_frac" : 0.01, "smooth_symm" : True, "sp_omega" : 1,
                "log_level" : 3, "max_levels" : 20}
ngs_pc = ngs_amg.AMG_H1(blf=a, freedofs=V.FreeDofs(), **ngs_amg_opts)
if comm.rank == 0:
    print(' --- NGS AMG LOGS ---')
    for a, b in ngs_pc.GetLogs().items():
        print(a, b)
    print(' --- NGS AMG LOGS ---')

#ngs_pc = c

ngs_pc = petsc.NGs2PETScPrecond(mat=mat_wrap, pc=ngs_pc)
ksp.SetPC(ngs_pc)

ksp.Finalize()

from time import time
t = -time()
gfu.vec.data = ksp * f.vec
t += time()

# t2 = -time()
# solvers.CG(mat=a.mat, rhs=f.vec, sol=gfu.vec, pre=ngs_pc, tol=1e-8, printrates=comm.rank==-1)
# t2 += time()

if comm.rank==0:
    print('TSOL P', t)
#    print('TSOL N', t2)

ksp_res = ksp.results
if comm.rank==0:
    print('ndof ', V.ndofglobal)
    print(' pc used: ', ksp_res['pc_used'])
    print('ksp converged? ', ksp_res['conv_r'])
    print('PETSc took nits:', ksp_res['nits'])
    print('init. norm res: ', ksp_res['errs'][0])
    print(' fin. norm res: ', ksp_res['res_norm'])

Draw(gfu, name='sol')

if ex_sol:
    err = f.vec.CreateVector()
    err.data = a.mat.Inverse(V.FreeDofs()) * f.vec - gfu.vec
    nr = Norm(err)
    if comm.rank==0:
        print('res ', nr)

