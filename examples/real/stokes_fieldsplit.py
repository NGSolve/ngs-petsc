from ngsolve import *
from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
mesh = Mesh( geo.GenerateMesh(maxh=0.07))

mesh.Curve(3)

V = VectorH1(mesh,order=3, dirichlet="wall|cyl|inlet")
Q = H1(mesh,order=2)

X = FESpace([V,Q])

u,p = X.TrialFunction()
v,q = X.TestFunction()

stokes = 1e-3*InnerProduct(grad(u), grad(v))+div(u)*q+div(v)*p - 1e-10*p*q
a = BilinearForm(X)
a += stokes * dx
a.Assemble()

# nothing here ...
f = LinearForm(X)   
f.Assemble()

# gridfunction for the solution
gfu = GridFunction(X)

# parabolic inflow at inlet:
uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))

# solve Stokes problem for initial conditions:
import ngs_petsc as petsc
petsc.Initialize()
opts = {"ksp_type" : "gmres",
        "ksp_atol" : 1e-30,
        "ksp_rtol" : 1e-14,
        "ksp_max_its" : 1e6,
        "ksp_monitor" : "",
        "ksp_converged_reason" : "",
        #"ksp_view" : "",
        "pc_type" : "none"}
# pmat = petsc.FlatPETScMatrix(a.mat, X.FreeDofs())
pmat = petsc.PETScMatrix(a.mat, X.FreeDofs())
inv_stokes = petsc.KSP(mat=pmat, name="ngs", petsc_options=opts, finalize=False)

fs_opts = {"pc_fieldsplit_detect_saddle_point" : ""}
# fs_opts = {"pc_fieldsplit_detect_saddle_point" : "", "fieldsplit_1_pc_type" : "jacobi"}
# fs_opts = {"pc_fieldsplit_schur_fact_type" : "full", "fieldsplit_pfield_pc_type" : "bjacobi"}
# fs_opts = dict()
fs_opts = { "pc_fieldsplit_type" : "schur",
            "pc_fieldsplit_schur_fact_type" : "diag",
            # "pc_fieldsplit_detect_saddle_point" : "",
            "pc_fieldsplit_schur_precondition" : "selfp",
            "fieldsplit_0_pc_type" : "jacobi",
            # "fieldsplit_0_ksp_monitor" : "",
            #"fieldsplit_0_ksp_converged_reason" : "",
            "fieldsplit_1_pc_type" : "jacobi",
            # "fieldsplit_1_ksp_monitor" : "",
            #"fieldsplit_1_ksp_converged_reason" : "",
            "fieldsplit_1_ksp_type" : "cg"}
pc = petsc.FieldSplitPrecond(pmat, "fspc", petsc_options=fs_opts)
pc.AddField(0,      V.ndof, "0")
pc.AddField(V.ndof, X.ndof, "1")
print('pc.finalize')
pc.Finalize()
print('pc.finalize done')
inv_stokes.SetPC(pc)

inv_stokes.Finalize()
res = f.vec.CreateVector()
res.data = f.vec - a.mat*gfu.vec
gfu.vec.data = inv_stokes * res

print('ndof V', V.ndof, 'free', sum(V.FreeDofs()))
print('ndof Q', Q.ndof, 'free', sum(Q.FreeDofs()))
print('ndof X', X.ndof, 'free', sum(X.FreeDofs()))

ksp_res = inv_stokes.results
if mpi_world.rank==0:
    print('ndof ', V.ndofglobal, Q.ndofglobal, X.ndofglobal)
    print(' pc used: ', ksp_res['pc_used'])
    print('ksp converged? ', ksp_res['conv_r'])
    print('PETSc took nits:', ksp_res['nits'])
    print('init. norm res: ', ksp_res['errs'][0])
    print(' fin. norm res: ', ksp_res['res_norm'])

Draw (Norm(gfu.components[0]), mesh, "velocity", sd=3)


ex_sol = True
if ex_sol:
    err = f.vec.CreateVector()
    err.data = a.mat.Inverse(X.FreeDofs()) * res - gfu.vec
    nr = Norm(err)
    if mpi_world.rank==0:
        print('res ', nr)

