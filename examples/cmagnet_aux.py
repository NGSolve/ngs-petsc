from ngsolve import *
from netgen.csg import *
import time

maxh = 0.5
diri = "outer"
order = 3
condense = True

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


import ngs_petsc as petsc
petsc.Initialize()
comm = mpi_world

ngsglobals.msg_level = 0
geom = MakeGeometry()
if comm.rank==0:
    ngmesh = geom.GenerateMesh(maxh=maxh)
    ngmesh.Distribute(comm)
else:
    from netgen.meshing import Mesh as NGMesh
    ngmesh = NGMesh.Receive(comm)
    ngmesh.SetGeometry(geom)
mesh = Mesh(ngmesh)
ngsglobals.msg_level = 0
mesh.Curve(3)
ngsglobals.msg_level = 3


### Set up the HCurl problem

with TaskManager(pajetrace = 10 * 1024 * 1024 if comm.rank in [0,1] else 0):

    HC = HCurl(mesh, order=order, dirichlet=diri)

    mur = { "core" : 1, "coil" : 1, "air" : 1 }
    mu0 = 1.257e-6
    nu_coef = [ 1/(mu0*mur[mat]) for mat in mesh.GetMaterials() ]
    nu = CoefficientFunction(nu_coef)
    alpha = nu
    beta = 1e-6 * nu
    sigma, tau = HC.TnT()
    a = BilinearForm(HC, condense=condense)
    a += SymbolicBFI(alpha * curl(sigma) * curl(tau) + beta * sigma * tau)
    jac = Preconditioner(a, "local")
    a.Assemble()

    f = LinearForm(HC)
    f += SymbolicLFI(CoefficientFunction((y,0.05-x,0)) * tau, definedon=mesh.Materials("coil"))
    f.Assemble()

    ### Set up the Gradient Space portion of the Preconditioner

    ## Gradient matrix and coresproding H1 space
    G, H1s = HC.CreateGradient()
    if comm.size > 1:
        G = ParallelMatrix(G, row_pardofs = H1s.ParallelDofs(), col_pardofs = HC.ParallelDofs(),
                           op=ParallelMatrix.C2C)

    us, vs = H1s.TnT()
    h1s_blf = BilinearForm(H1s, condense=condense)
    h1s_blf += beta * grad(us) * grad(vs) * dx
    h1s_blf.Assemble()


    def SelectH1sPC (var = 2, nullspace = False):
        #######
        # Give the vector representation of the constant function to the PETSc-Matrix.
        # This allows AMG preconditioners to identify the low-order part
        if var in [0,1,2]:
            h1smat = petsc.PETScMatrix(h1s_blf.mat, H1s.FreeDofs(condense))
            if nullspace:
                cgf = h1s_blf.mat.CreateColVector()
                cgf[:] = 0
                cgf[0:H1s.lospace.ndof] = 1
                h1smat.SetNearNullSpace([cgf])
        ######
        if var == 0:
            # Solve the scalar Problem with Preconditioned CG
            return petsc.KSP(h1smat, "h1spc",
                             petsc_options={ "ksp_type" : "cg",
                                             #"ksp_view" : "",
                                             # "ksp_monitor" : "ascii:h1s_ksp",
                                             "ksp_converged_reason" : "",
                                             "pc_type" : "gamg"})
        elif var == 1:
            # Do a fixed number of Steps of Preconditioned CG
            return petsc.KSP(h1smat, "h1spc",
                             petsc_options={ "ksp_type" : "cg",
                                             "ksp_norm_type" : "none",
                                             "ksp_max_it" : 5,
                                             "pc_type" : "gamg"})
        elif var == 2:
            # Simple AMG preconditioner
            return petsc.PETSc2NGsPrecond(h1smat, "h1spc", petsc_options={"pc_type" : "gamg"})
        elif var == 10:
            # Exact Inverse
            return h1s_blf.mat.Inverse(H1s.FreeDofs(condense))

        raise 'invalid choice scal'

    ## Putting the gradient range preconditioner together
    pc_h1s = SelectH1sPC(2, True)
    pcgrad = G @ pc_h1s @ G.T


    ### Set up the Vector-H1 portion of the Preconditioner

    H1v = VectorH1(mesh, order=max(1, order), dirichlet=diri)

    ## H1->HCurl embedding
    if True:
        from fast_embed import FastEmbed
        E = FastEmbed(V_GOAL=HC, V_ORIGIN=H1v)
        ET = E.T
    else:
        hcmass = BilinearForm(HC)
        hcmass += sigma * tau * dx
        hcmass.Assemble()

        uv, vv = H1v.TnT()
        mixmass = BilinearForm(trialspace=H1v, testspace=HC)
        mixmass += uv * tau * dx
        mixmass.Assemble()

        hcm_inv = hcmass.mat.Inverse(HC.FreeDofs(), inverse="sparsecholesky" if comm.size==1 else "mumps")

        E = hcm_inv @ mixmass.mat
        ET = mixmass.mat.T @ hcm_inv

    ## Vector-H1 space preconditioner
    uv, vv = H1v.TnT()
    h1v_blf = BilinearForm(H1v, condense=condense)
    h1v_blf += alpha * InnerProduct(grad(uv), grad(vv)) * dx
    h1v_blf.Assemble()


    # Give the vector representation of constant functions to the PETSc-Matrix.
    # This allows AMG preconditioners to identify the low-order part
    # Additionally, it tells the AMG that we have three components 

    def SelectH1vPC (var = 2, nullspace = False):
        #######
        # Give the vector representation of the constant function to the PETSc-Matrix.
        # This allows AMG preconditioners to identify the low-order part
        if var in [0,1,2]:
            h1vmat = petsc.PETScMatrix(h1v_blf.mat, H1v.FreeDofs(condense))
            if nullspace:
                const_vecs = []
                for l in range(3):
                    v = h1v_blf.mat.CreateRowVector()
                    v[:] = 0
                    v[H1v.Range(l).start  : H1v.Range(l).start + H1v.components[l].lospace.ndof] = 1
                    const_vecs.append(v)
                h1vmat.SetNearNullSpace(const_vecs)
        ######
        if var == 0:
            # Solve the scalar Problem with Preconditioned CG
            return petsc.KSP(h1vmat, "h1vpc",
                             petsc_options={ "ksp_type" : "cg",
                                             # "ksp_monitor" : "ascii:h1s_ksp",
                                             # "ksp_converged_reason" : "ascii:h1s_ksp",
                                             "ksp_converged_reason" : "",
                                             "pc_type" : "gamg"})
        elif var == 1:
            # Do a fixed number of Steps of Preconditioned CG
            return petsc.KSP(h1vmat, "h1vpc",
                             petsc_options={ "ksp_type" : "cg",
                                             # "ksp_norm_type" : "none",
                                             "ksp_max_it" : 5,
                                             "pc_type" : "gamg"})
        elif var == 2:
            # Simple AMG preconditioner
            return petsc.PETSc2NGsPrecond(h1vmat, "h1vpc", petsc_options={"pc_type" : "gamg"})
        elif var == 10:
            # Exact Inverse
            return h1v_blf.mat.Inverse(H1v.FreeDofs(condense))

        raise 'invalid choice vec'

    ## Putting the vector-h1 preconditioner together
    pc_h1v = SelectH1vPC(2, True)
    pcvec = E @ pc_h1v @ ET

    gfu = GridFunction(HC)

    ### The smoother component
    pcsmo = jac.mat

    # hcmat = petsc.PETScMatrix(a.mat, HC.FreeDofs(condense))
    # pcsmo = petsc.PETSc2NGsPrecond(hcmat, "pcsmo", petsc_options = {"pc_type" : "sor"})

    # the full preconditioner
    pc = pcvec + pcgrad + pcsmo

    pam = petsc.FlatPETScMatrix(a.mat, HC.FreeDofs(condense))
    ksp = petsc.KSP(mat=pam, name="aux_ksp",
                    petsc_options = {"ksp_rtol" : 1e-6,
                                     "ksp_norm_type" : "preconditioned",
                                     "ksp_view_eigenvalues" : "",
                                     "ksp_atol" : 1e-50,
                                     "ksp_type" : "cg",
                                     "ksp_max_it" : 500,
                                     "ksp_monitor" : "",
                                     "ksp_converged_reason" : ""},
                    finalize=False)
    pcpc = petsc.ConvertNGsPrecond(pc, mat=pam, name="ngs_side_aux_pc")
    ksp.SetPC(pcpc)
    ksp.Finalize()


    comm.Barrier()
    t1 = -time.time()
    gfu.vec.data = ksp * f.vec
    t1 = t1 + time.time()

    comm.Barrier()
    t2 = -time.time()
    solvers.CG(mat=a.mat, pre=pc, rhs=f.vec, sol=gfu.vec, tol=1e-6, maxsteps=500, printrates=mpi_world.rank==0)
    t2 = t2 + time.time()    


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


    ex_sol = True
    if ex_sol:
        err = f.vec.CreateVector()
        exsol = f.vec.CreateVector()

        exsol.data = a.mat.Inverse(HC.FreeDofs(condense)) * f.vec
        err.data = exsol - gfu.vec
        nerr = Norm(err)

        if comm.rank==0:
            print('err ', nerr)
