#
#
#  2D Linear Elasticity, KSP+AMG (including near null-space)
#
#
# We consider 2 formulations for linear elasticity in 2D.
#    - displacement only
#    - displacement+rotation
#
# For FEM-Spaces we use:
#  - Compound/VectorFESpaces:
#     all DOFs belonging to the same physical quantity are
#     enumerated in series (u^x_1, u^x_2, ...; u^y_1, u^y_2,...)
#       -> uses PETSc MPIAIJ-mats
#  - Multidim-FESpaces:
#     all DOFs belonging to the same node are enumerated in series
#     (u^x_1, u^y_1,r_1; u^x_2, u^y_2,r_2; ...)
#       -> uses PETSc MPIBAIJ-mats
#
# We use PCG with an AMG preconditioner:
#  - Without modifications
#  - We give the rigid body modes to the AMG solver
#    to improve convergence
#
#


from ngsolve import *
from netgen.geom2d import *
import time, sys, os
import ngspetsc as petsc
from netgen.meshing import Mesh as NGMesh
from ngsolve.meshes import MakeStructured2DMesh

# master meshes, then we distribute
def ParMesh(mesh_func):
    comm = MPI_Init()
    ngmesh = NGMesh(dim=2)
    if comm.rank==0:
        ngmesh = mesh_func()
    comm.Barrier()
    ngmesh.Distribute(comm)
    return Mesh(ngmesh)
    
# geom 1: BEAM
# a 10x1 beam, fixed at the left side
def mesh_beam(maxh):
    geo = SplineGeometry()
    geo.AddRectangle((0, -1), (10,1), leftdomain=1, rightdomain=0, bcs = ("bottom", "right", "top", "left"))
    mesh = geo.GenerateMesh(maxh=maxh)
    return mesh


# geom 2: STRIP
# a very long strip (difficult for AMG!)
def mesh_strip(Lf):
    #workaround: set NGS-comm to local comm
    comm = MPI_Init()
    sc = comm.SubComm([0])
    from ngsolve.ngstd import SetNGSComm
    SetNGSComm(sc)
    ngs_mesh = MakeStructured2DMesh(quads=False, nx=Lf, ny=1, mapping=lambda x,y : (Lf*x,y))
    mesh = ngs_mesh.ngmesh
    SetNGSComm(comm)
    return mesh

# geom 3: FIBERS
# a 10x1 beam, fixed at the left side
# consists of many thin layers with different material properties
# also difficult for the solver!
def MyMakeRectangle (geo, p1, p2, bc=None, bcs=None, rightdomain=[0,0,0,0], **args):
    p1x, p1y = p1
    p2x, p2y = p2
    p1x,p2x = min(p1x,p2x), max(p1x, p2x)
    p1y,p2y = min(p1y,p2y), max(p1y, p2y)
    if not bcs: bcs=4*[bc]
    pts = [geo.AppendPoint(*p) for p in [(p1x,p1y), (p2x, p1y), (p2x, p2y), (p1x, p2y)]]
    for p1,p2,bc,rd in [(0,1,bcs[0], rightdomain[0]), (1, 2, bcs[1], rightdomain[1]), (2, 3, bcs[2], rightdomain[2]), (3, 0, bcs[3], rightdomain[3])]:
        geo.Append( ["line", pts[p1], pts[p2]], bc=bc, rightdomain=rd, **args)
def make_fiber_geo(n_fibers = 3, fiber_box = [10,1], fiber_rad = 0.25, maxh_mat = 0.5, maxh_fiber = 0.3):
    geo = SplineGeometry()
    geo.SetMaterial(1, "mat")
    geo.SetMaterial(2, "fiber")
    xmax = fiber_box[0]
    # Points
    y = n_fibers * fiber_box[1]
    geo.AppendPoint(0, y)
    geo.AppendPoint(xmax, y)
    # top points:
    for n in range(n_fibers):
        #top fiber
        y = y - (0.5*fiber_box[1]-fiber_rad)
        geo.AppendPoint(0, y)
        geo.AppendPoint(xmax, y)
        y = y - 2*fiber_rad
        #bot fiber
        geo.AppendPoint(0, y)
        geo.AppendPoint(xmax, y)
        #top next fiber
        y = y - (0.5*fiber_box[1]-fiber_rad)
    #bot box
    geo.AppendPoint(0, y)
    geo.AppendPoint(xmax, y)
    #fibers, between, top+bot
    n_layers = n_fibers + (n_fibers-1) + 2
    n_points = 2 * (n_layers+1)
    # Segments
    # top seg
    geo.Append( ["line", 0, 1], bc="top", leftdomain=0, rightdomain=1)
    # bot seg
    geo.Append( ["line", n_points-2, n_points-1], bc="bot", leftdomain=1, rightdomain=0)
    # interface segs
    ld = 1
    for k in range(n_layers-1):
        geo.Append( ["line", 2+2*k, 2+2*k+1], bc="interface", leftdomain=ld, rightdomain=3-ld)
        ld = 3-ld
    # left/right segs
    ind = 1
    for k in range(n_layers):
        geo.Append( ["line", 2*k, 2*(k+1)], bc="left", leftdomain=ind, rightdomain=0)
        geo.Append( ["line", 1+2*k, 1+2*(k+1)], bc="right", leftdomain=0, rightdomain=ind)
        ind = 3-ind
    return geo
def mesh_fiber(n_fibers):
    yd = 2
    r = yd/(4.0*n_fibers)
    geo = make_fiber_geo(n_fibers=n_fibers, fiber_box=[5,4*r], fiber_rad=r)
    mesh = geo.GenerateMesh(maxh=2*r)
    return mesh

# formulation with rotation
def setup_rots(mesh, comp=True):
    if comp:
        fes1 = VectorH1(mesh, order=1, dirichlet="left")
        fes2 = H1(mesh, order=1, dirichlet="left")
        fes = FESpace( [fes1,fes2] )
        (u,w),(ut,wt)  = fes.TnT()
        gradu = grad(u)
        gradut = grad(ut)
    else:
        fes = H1(mesh, order=1, dirichlet="left", dim=3)
        U,V = fes.TnT()
        u = CoefficientFunction((U[0], U[1]))
        w = U[2]
        ut = CoefficientFunction((V[0], V[1]))
        wt = V[2]
        gradu = CoefficientFunction((grad(U)[0,0],grad(U)[0,1],grad(U)[1,0],grad(U)[1,1]), dims=[2,2])
        gradut = CoefficientFunction((grad(V)[0,0],grad(V)[0,1],grad(V)[1,0],grad(V)[1,1]), dims=[2,2])
    wmat = CoefficientFunction( (0, w, -w, 0), dims = (2,2) )
    wtmat = CoefficientFunction( (0, wt, -wt, 0), dims = (2,2) )
    factor = {"fiber":1e4, "mat":1, "default":1}
    cf_factor = CoefficientFunction( [ factor[mat] for mat in mesh.GetMaterials() ] )
    coef = cf_factor
    force = CoefficientFunction( (0, -0.0002) )
    a = BilinearForm(fes, symmetric=False)
    a += SymbolicBFI( InnerProduct(coef*(gradu - wmat), gradut - wtmat))
    f = LinearForm(fes)
    f += SymbolicLFI( force*ut )
    a.Assemble()
    f.Assemble()
    return fes, a, f

# formulation without rotation
def setup_norots(mesh, comp=True):
    if comp:
        fes = VectorH1(mesh, order=1, dirichlet="left")
    else:
        fes = H1(mesh, order=1, dirichlet="left", dim=2)
    u,v = fes.TnT()
    gradu = grad(u)
    epsu = 0.5 * (gradu + gradu.trans)
    gradv = grad(v)
    epsv = 0.5 * (gradv + gradv.trans)
    factor = {"fiber":1e4, "mat":1, "default":1}
    cf_factor = CoefficientFunction( [ factor[mat] for mat in mesh.GetMaterials() ] )
    coef = cf_factor
    force = CoefficientFunction( (0, -0.0002) )
    a = BilinearForm(fes, symmetric=False)
    a += SymbolicBFI( InnerProduct(coef*epsu, epsv))
    f = LinearForm(fes)
    f += SymbolicLFI( force*v )
    a.Assemble()
    f.Assemble()
    return fes, a, f
        
# setup the FEM formulation
def setup(mesh, comp=True, rots=True):
    if rots:
        return setup_rots(mesh, comp)
    else:
        return setup_norots(mesh, comp)

def rb_modes(fes, comp=False, rots=False):
    RBMS = [(1,0,0), (0,1,0), (y, -x, 1)]
    gfu = GridFunction(fes)
    rbm_vecs = list()
    upart = gfu.components[0] if comp and rots else gfu
    rpart = gfu.components[1] if comp and rots else None
    for RBM in RBMS:
        if comp or not rots:
            ucf = CoefficientFunction((RBM[0], RBM[1]))
            upart.Set(ucf)
        if comp and rots:
            rcf = CoefficientFunction(RBM[2])
            rpart.Set(rcf)
        if rots and not comp:
            gfu.Set(CoefficientFunction(RBM))
        v = gfu.vec.CreateVector()
        v.data = gfu.vec
        rbm_vecs.append(v)
    return rbm_vecs

# C++ timers are not reset between calls, so we use tsol, tsup
# to keep track
def TestKSP(fes, a, f, tsol, tsup, kvecs=list(), vfac=1):
    comm = MPI_Init()
    gfu = GridFunction(fes)
    # calculare residuum to confirm correctness of KSP-solve
    P = Projector(fes.FreeDofs(), True)
    gfu.vec[:] = 0
    res = f.vec.CreateVector()
    p_res = f.vec.CreateVector()
    res.data = f.vec
    res.data -= a.mat * gfu.vec
    res.Cumulate()
    p_res.local_vec.data = P * res.local_vec
    nr_init = Norm(p_res)
    if comm.rank==0:
        print('------- init. norm res ', nr_init)
    petsc.KSPSolve(blf=a, rhs=f.vec, sol=gfu.vec, fds=fes.FreeDofs(), kvecs=kvecs)
    ts = {t['name'] : t for t in Timers()}
    t_ksp_sup = comm.Max(ts['KSP - setup']['time']) - tsup
    tsup += t_ksp_sup
    t_ksp_sol = comm.Max(ts['KSP - solve']['time']) - tsol
    tsol += t_ksp_sol
    if comm.rank==0:
        nvg = vfac * fes.ndofglobal
        print('KSP setup: ', t_ksp_sup)
        print('KSP solve nV/(t*NP): ', nvg/(t_ksp_sup*comm.size))
        print('KSP solve: ', t_ksp_sol)
        print('KSP solve nV/(t*NP): ', nvg/(t_ksp_sol*comm.size))
    res = f.vec.CreateVector()
    p_res = f.vec.CreateVector()
    res.data = f.vec
    res.data -= a.mat * gfu.vec
    res.Cumulate()
    p_res.local_vec.data = P * res.local_vec
    nr = Norm(p_res)
    if comm.rank==0:
        print('------- fin.  norm res ', nr)
        print('------- rel. norm res ', nr/nr_init)
    return tsol, tsup

def test_case(tsol, tsup, mesh_func, title="unnamed"):
    comm = MPI_Init()
    mesh = ParMesh(mesh_func)
    for C,CN in [(True, '_comp'), (False,'_mdim')]:
        for R, RN in [(True, '_rots'), (False,'_norots')]:
            fes, a, f = setup(mesh, comp=C, rots=R)
            kvecs = rb_modes(fes, comp=C, rots=R)
            nd_to_v = 1/3 if R else 1/2
            nd_to_v = nd_to_v if C else 1
            nd_to_row = 3 if R else 2
            nd_to_row = 1 if C else nd_to_row
            if comm.rank==0:
                print('\n---------------------')
                print('KSP for '+title+CN+RN)
                print('glob ND: ', fes.ndofglobal)
                print('glob nrows: ', nd_to_row*fes.ndofglobal)
            TestKSP(fes, a, f, tsol, tsup, vfac=nd_to_v)
            if comm.rank==0:
                print('---------------------')
                print('---------------------')
                print('KSP + RBM for '+title+CN+RN)
                print('glob ND: ', fes.ndofglobal)
                print('glob nrows: ', nd_to_row*fes.ndofglobal)
            TestKSP(fes, a, f, tsol, tsup, kvecs, vfac=nd_to_v)
            if comm.rank==0:
                print('---------------------\n')
    return tsol, tsup

if __name__=='__main__':
    #meshfile = 'strips/strip_LF4_lay1.vol'
    meshfile = 'fibers1/2d_50fibers.vol'
    ngsglobals.msg_level = 1
    comm = MPI_Init()
    petsc.InitPETSC()
    tsol, tsup = 0,0
    tsol, tsup = test_case(tsol, tsup, title = "BEAM", mesh_func = lambda : mesh_beam(0.1)) # maxh=0.1
    tsol, tsup = test_case(tsol, tsup, title = "STRIP", mesh_func = lambda : mesh_strip(int(1e2))) # 1x100 strip
    tsol, tsup = test_case(tsol, tsup, title = "FIBER", mesh_func = lambda : mesh_fiber(15)) # 15 fibers

    
