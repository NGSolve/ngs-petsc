from ngsolve import *
import ngs_petsc as petsc
from netgen.meshing import Mesh as NGMesh

def make_geo():
    import netgen.geom2d as geom2d
    geo = geom2d.SplineGeometry()
    pnums = [ geo.AddPoint (x,y,maxh=0.01) for x,y in [(0,0), (1,0), (1,0.1), (0,0.1)] ]
    for p1,p2,bc in [(0,1,"bot"), (1,2,"right"), (2,3,"top"), (3,0,"left")]:
        geo.Append(["line", pnums[p1], pnums[p2]], bc=bc)
    return geo

geo = make_geo()

comm = mpi_world
if comm.rank == 0:
    mesh = geo.GenerateMesh(maxh=0.025)
    if comm.size > 1:
        mesh.Distribute(comm)
else:
    mesh = NGMesh.Receive(comm)
mesh = Mesh(mesh)

# E module and poisson number:
E, nu = 210, 0.2
# Lamé constants:
mu  = E / 2 / (1+nu)
lam = E * nu / ((1+nu)*(1-2*nu))

V = H1(mesh, order=1, dirichlet="left", dim=mesh.dim)
u  = V.TrialFunction()

#gravity:
force = CoefficientFunction( (0,-1) )



def Pow(a, b):
    return exp (log(a)*b)

def NeoHook (C):
    return 0.5 * mu * (Trace(C-I) + 2*mu/lam * Pow(Det(C), -lam/2/mu) - 1)

I = Id(mesh.dim)
F = I + u.Deriv()   # attention: row .. component, col .. derivative
C = F * F.trans

factor = Parameter(1.0)

a = BilinearForm(V, symmetric=False)
a += SymbolicEnergy(  NeoHook (C).Compile(False) )
a += SymbolicEnergy(  (-factor * InnerProduct(force,u) ).Compile(False) )

gfu = GridFunction(V)
gfu2 = GridFunction(V)

poo = dict()
#poo = {"info" : ""}
petsc.Initialize(**poo)

petsc_options = {"ksp_pc_type" : "none",
                 "ksp_type" : "cg",
                 "ksp_atol" : 1e-30,
                 "ksp_rtol" : 1e-8,
                 "ksp_max_it" : 1000,
                 #"ksp_monitor" : "",
                 #"info" : "",
                 #"ksp_converged_reason" : "",
                 #"snes_monitor" : "",
                 #"snes_view" : "",
                 "snes_converged_reason" : "",
                 "snes_max_it" : 50,
                 "snes_linesearch_type" : "basic" }
snes = petsc.SNES(a, name="mysnes", petsc_options=petsc_options)
snes_ksp = snes.GetKSP()

res1 = gfu.vec.CreateVector()
res2 = gfu.vec.CreateVector()



def SolveNonlinearMinProblem(a,gfu,tol=1e-13,maxits=25):
    res = gfu.vec.CreateVector()
    du  = gfu.vec.CreateVector()

    for it in range(maxits):
        # print ("Newton iteration {:3}".format(it),end="\n")
        # print ("energy = {:16}".format(a.Energy(gfu.vec)),end="\n")

        #solve linearized problem:
        a.Apply (gfu.vec, res)
        a.AssembleLinearization (gfu.vec)
        inv = a.mat.Inverse(V.FreeDofs())
        du.data = inv * res

        #update iteration
        gfu.vec.data -= du

        #stopping criteria
        stopcritval = sqrt(abs(InnerProduct(du,res)))
        # print ("<A u",it,", A u",it,">_{-1}^0.5 = ", stopcritval)
        if stopcritval < tol:
            if comm.rank == 0:
                print("NGS-newton after ", it)
            break
        # Redraw(blocking=True)


import sys
Draw (mesh, deformation=gfu,  name="u - ptc")
Draw (mesh, deformation=gfu2, name="u - ngs")
ngsglobals.msg_level = 0
for loadstep in range(50):
    print ("Solve loadstep", loadstep)
    sys.stdout.flush()
    factor.Set ((loadstep+1)/10)

    snes.Solve(gfu.vec)
    SolveNonlinearMinProblem(a,gfu2)


    res1.data = gfu.vec - gfu2.vec
    rn = Norm(res1)
    if comm.rank == 0:
        print("diff:", rn)

    gfu.vec.Cumulate()
    a.Apply(gfu.vec, res1)
    rn = Norm(res1)
    if comm.rank == 0:
        print("p - res norm:", rn)

    a.Apply(gfu2.vec, res2)
    rn = Norm(res2)
    if comm.rank == 0:
        print("n - res norm:", rn)

    Redraw()

    ksp_res = snes_ksp.results
    # if comm.rank==0:
    #     # print(' ----- ')
    #     # for k,v in ksp_res.items():
    #     #     print(k, v)
    #     print(' ----- ')
    #     print('ndof ', V.ndofglobal)
    #     print(' pc used: ', ksp_res['pc_used'])
    #     print('ksp converged? ', ksp_res['conv_r'])
    #     print('PETSc took nits:', ksp_res['nits'])
    #     print('init. norm res: ', ksp_res['errs'][0])
    #     print(' fin. norm res: ', ksp_res['res_norm'])
    #     print(' ----- ')

        
