from ngsolve import *
from netgen.csg import *
import time

import ngs_petsc as petsc
petsc.Initialize()

comm = mpi_world

def MakeGeometry():
    geometry = CSGeometry()
    box = OrthoBrick(Pnt(-1,-1,-1),Pnt(2,1,2)).bc("outer")

    core = OrthoBrick(Pnt(0,-0.05,0),Pnt(0.8,0.05,1))- \
           OrthoBrick(Pnt(0.1,-1,0.1),Pnt(0.7,1,0.9))- \
           OrthoBrick(Pnt(0.5,-1,0.4),Pnt(1,1,0.6)).mat("core")
    
    coil = (Cylinder(Pnt(0.05,0,0), Pnt(0.05,0,1), 0.3) - \
            Cylinder(Pnt(0.05,0,0), Pnt(0.05,0,1), 0.15)) * \
            OrthoBrick (Pnt(-1,-1,0.3),Pnt(1,1,0.7)).mat("coil")
    
    geometry.Add ((box-core-coil).mat("air"))
    geometry.Add (core)
    geometry.Add (coil)
    return geometry

geom = MakeGeometry()
if comm.rank==0:
    ngmesh = geom.GenerateMesh(maxh=0.5)
    ngmesh.Distribute(comm)
else:
    from netgen.meshing import Mesh as NGMesh
    ngmesh = NGMesh.Receive(comm)
    ngmesh.SetGeometry(geom)
ngmesh.Refine()
mesh = Mesh(ngmesh)
mesh.Curve(5)

ngsglobals.msg_level = 0

diri = "outer"

order = 0
HC = HCurl(mesh, order=order, dirichlet=diri)
sigma, tau = HC.TnT()


mur = { "core" : 1000, "coil" : 1, "air" : 1 }
mu0 = 1.257e-6
nu_coef = [ 1/(mu0*mur[mat]) for mat in mesh.GetMaterials() ]
nu = CoefficientFunction(nu_coef)
alpha = nu
beta = 1e-6 * nu
a = BilinearForm(HC, symmetric=False)
a += SymbolicBFI(alpha * curl(sigma) * curl(tau) + beta * sigma * tau)
jac = Preconditioner(a, "local")
a.Assemble()


## HCurl problem
f = LinearForm(HC)
f += SymbolicLFI(CoefficientFunction((0,0,1)) * tau, definedon=mesh.Materials("coil"))
f.Assemble()


## Gradient matrix
G, H1s = HC.CreateGradient()
us, vs = H1s.TnT()


## H1->HCurl embedding
H1v = VectorH1(mesh, order=order, dirichlet=diri)
uv, vv = H1v.TnT()

hcmass = BilinearForm(HC)
hcmass += sigma * tau * dx
hcmass.Assemble()

mixmass = BilinearForm(trialspace=H1v, testspace=HC)
mixmass += uv * tau * dx
mixmass.Assemble()

hcm_inv = hcmass.mat.Inverse(HC.FreeDofs(), inverse="sparsecholesky")

E = hcm_inv @ mixmass.mat


## H1 scalar problem
h1scal_blf = BilinearForm(H1s)
h1scal_blf += beta * grad(us) * grad(vs) * dx
h1scal_blf.Assemble()

h1smat = petsc.PETScMatrix(h1scal_blf.mat, H1s.FreeDofs())
# pc_h1s = h1scal_blf.mat.Inverse(H1s.FreeDofs())
# pc_h1s = petsc.KSP(h1smat, "h1spc", petsc_options={ "ksp_type" : "cg",
#                                                     #"ksp_monitor" : "",
#                                                     #"ksp_converged_reason" : "",
#                                                     "pc_type" : "gamg"})
pc_h1s = petsc.PETSc2NGsPrecond(h1smat, "h1spc", petsc_options={"pc_type" : "gamg"})


## H1 vector problem
h1v_blf = BilinearForm(H1v)
h1v_blf += alpha * InnerProduct(grad(uv), grad(vv)) * dx
h1v_blf.Assemble()

h1vmat = petsc.PETScMatrix(h1v_blf.mat, H1v.FreeDofs())
# h1v_blf.mat.Inverse(H1v.FreeDofs())
# pc_h1v = petsc.KSP(h1vmat, "h1vpc", petsc_options = { "ksp_converged_reason" : "",
#                                                       "pc_type" : "gamg"})
pc_h1v = petsc.PETSc2NGsPrecond(h1vmat, name="h1vpc", petsc_options = { "pc_type" : "gamg" })


# gradient range preconditioner
pcgrad = G @ pc_h1s @ G.T

# vector-h1 preconditioner
pcvec = E @ pc_h1v @ E.T

# the full preconditioner
pc = pcgrad + pcvec + jac.mat

gfu = GridFunction(HC)
t1 = -time.time()
solvers.CG(mat=a.mat, pre=pc, rhs=f.vec, sol=gfu.vec, tol=1e-6, maxsteps=500, printrates=mpi_world.rank==0)
t1 = t1 + time.time()

# from ngsolve.la import EigenValues_Preconditioner
# ngsglobals.msg_level = 0
# evs = EigenValues_Preconditioner(a.mat, pc)
# print('min, max ev = ', evs[0], '/', evs[len(evs)-1], ', Condition = ', evs[len(evs)-1]/evs[0])


if comm.rank == 0:
    print(' ----------- ')
    print('ndof H-Curl space: ', HC.ndofglobal)
    print('low order ndof H-Curl space: ', HC.lospace.ndofglobal)
    print('t solve', t1)
    print('dofs / (sec * np) ', HC.ndofglobal / (t1 * max(comm.size-1, 1)) )
    print(' ----------- ')

ex_sol = False
if ex_sol:
    err = f.vec.CreateVector()
    exsol = f.vec.CreateVector()

    exsol.data = a.mat.Inverse(HC.FreeDofs()) * f.vec
    err.data = exsol - gfu.vec
    nerr = Norm(err)

    if comm.rank==0:
        print('err ', nerr)
