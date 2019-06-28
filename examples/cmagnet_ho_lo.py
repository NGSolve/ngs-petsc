from netgen.csg import *

from ngsolve import *

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

ngsglobals.msg_level = 0

geom = MakeGeometry()
if comm.rank==0:
    ngmesh = geom.GenerateMesh(maxh=0.5)
    ngmesh.Distribute(comm)
else:
    from netgen.meshing import Mesh as NGMesh
    ngmesh = NGMesh.Receive(comm)
    ngmesh.SetGeometry(geom)
mesh = Mesh(ngmesh)

ngsglobals.msg_level = 0

mesh.Curve(5)

ngsglobals.msg_level = 1

HC = HCurl(mesh, order=5, dirichlet="outer", nograds = True)
u,v = HC.TnT()

a = BilinearForm(HC, symmetric=False)
mur = { "core" : 1000, "coil" : 1, "air" : 1 }
mu0 = 1.257e-6
nu_coef = [ 1/(mu0*mur[mat]) for mat in mesh.GetMaterials() ]
nu = CoefficientFunction(nu_coef)
a += SymbolicBFI(nu*curl(u)*curl(v) + 1e-6*nu*u*v)

# c = Preconditioner(a, "petsc_pc_hypre_ams")
c = Preconditioner(a, "bddc", coarsetype="petsc_pc_hypre_ams", petsc_pc_petsc_options=["pc_hypre_ams_print_level 3"])
# c = Preconditioner(a, "bddc")


f = LinearForm(HC)
f += SymbolicLFI(CoefficientFunction((y,0.05-x,0)) * v, definedon=mesh.Materials("coil"))

ngsglobals.numthreads = 4 if comm.size == 1 else 1
# with TaskManager():
#     f.Assemble()
#     a.Assemble()

f.Assemble()
a.Assemble()

# amat = petsc.PETScMatrix(a.mat, row_freedofs=HC.FreeDofs(), col_freedofs=HC.FreeDofs())
amat = petsc.FlatPETScMatrix(a.mat, row_freedofs=HC.FreeDofs(), col_freedofs=HC.FreeDofs())
ksp = petsc.KSP(amat, name="cmagnet",
                petsc_options = { "ksp_type" : "cg",
                                  "ksp_atol" : 1e-30,
                                  "ksp_monitor" : "",
                                  # "ksp_view" : "",
                                  # "ksp_view_eigenvalues" : "" },
                                  "ksp_rtol" : 1e-6},
                finalize = False)
ksp.SetPC(petsc.ConvertNGsPrecond(c, mat=amat))
ksp.Finalize()

from time import time

gfu = GridFunction(HC)

comm.Barrier()
t1 = -time()
gfu.vec.data = ksp * f.vec
comm.Barrier()
t1 = t1 + time()

comm.Barrier()
t2 = -time()
solvers.CG(mat=a.mat, pre=c, rhs=f.vec, tol=1e-6, printrates=comm.rank==0)
comm.Barrier()
t2 = t2 + time()

if comm.rank == 0:
    print(' ----------- ')
    print('ndof H-Curl space: ', HC.ndofglobal)
    print('low order ndof H-Curl space: ', HC.lospace.ndofglobal)
    print(' --- KSP --- ')
    print('t solve', t1)
    print('dofs / (sec * np) ', HC.ndofglobal / (t1 * max(comm.size-1, 1)) )
    print(' --- NGs-CG --- ')
    print('t solve', t2)
    print('dofs / (sec * np) ', HC.ndofglobal / (t2 * max(comm.size-1, 1)) )
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

Draw (gfu.Deriv(), mesh, "B-field", draw_surf=False)

