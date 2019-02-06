from ngsolve import *
import ngspetsc as petsc

def make_mesh():
    comm = MPI_Init()
    from netgen.meshing import Mesh as NGMesh
    ngm = NGMesh(dim=2)
    if comm.rank==0:
        from netgen.geom2d import unit_square
        ngm = unit_square.GenerateMesh(maxh=0.01)
    ngm.Distribute()
    return Mesh(ngm)


comm = MPI_Init()
mesh = make_mesh()
V = H1(mesh, order=1, dirichlet='.*')
u,v = V.TnT()
a = BilinearForm(V)
a += SymbolicBFI(grad(u)*grad(v))
a.Assemble()
f = LinearForm(V)
f += SymbolicLFI(v)
f.Assemble()
gfu = GridFunction(V)

opts = {"ksp_type":"cg", "ksp_atol":"1e-30", "ksp_rtol":"1e-14", "pc_type":"gamg"}
ksp_res = petsc.KSPSolve(blf=a, rhs=f.vec, sol=gfu.vec, fds=V.FreeDofs(), **opts)
if comm.rank==0:
    print('PETSc took nits:', ksp_res['nits'])
    print('init. norm res: ', ksp_res['errs'][0])
    print(' fin. norm res: ', ksp_res['res_norm'])

Draw(gfu, name='sol')
