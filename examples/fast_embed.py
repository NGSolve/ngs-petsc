from ngsolve import *
import ngsolve
from ngsolve.ngstd import Timer

class FastInv(BaseMatrix):
    def __init__ (self, mat, blocks1, blocks2):
        super(FastInv, self).__init__()
        self.parmat = mat
        self.spmat = self.parmat if mpi_world.size == 1 else self.parmat.local_mat
        self.inv1 = self.spmat.CreateBlockSmoother(blocks1)
        self.inv2 = self.spmat.CreateBlockSmoother(blocks2)
        self.res = self.spmat.CreateColVector()
        self.ty = self.spmat.CreateColVector()
        self.tmult = Timer("FastInv")
        
    def Height(self):
        return self.parmat.height

    def Width(self):
        return self.parmat.width

    def CreateRowVector(self):
        return self.spmat.CreateRowVector()

    def CreateColVector(self):
        return self.spmat.CreateColVector()

    def MultAdd(self, scal, x, y):
        self.ty[:] = 0
        self.Mult(x, self.ty)
        y.local_vec.data += scal * self.ty
        
    def Mult(self, x, y):
        self.tmult.Start()
        y.local_vec.data = self.inv1 * x.local_vec
        self.res.data = x - self.spmat * y
        y.local_vec.data += self.inv2 * self.res
        self.tmult.Stop()
        
    def MultTransAdd(self, scal, x, y):
        self.ty[:] = 0
        self.MultTrans(x, self.ty)
        y.local_vec.data += scal * self.ty

    def MultTrans(self, x, y):
        self.tmult.Start()
        y.local_vec.data = self.inv2.T * x.local_vec
        self.res.data = x - self.spmat.T * y
        y.local_vec.data += self.inv1.T * self.res
        self.tmult.Stop()

    def IsComplex(self):
        return False

def FastEmbed(V_GOAL, V_ORIGIN):

    sigma, tau = V_GOAL.TnT()
    taudual = tau.Operator("dual")
    u, v = V_ORIGIN.TnT()

    amix = BilinearForm(trialspace=V_ORIGIN, testspace=V_GOAL)
    amix += u * taudual * dx(element_vb=BBND)
    amix += u * taudual * dx(element_vb=BND)
    # amix += u * taudual * dx
    amix.Assemble()

    agoal = BilinearForm(V_GOAL)
    agoal += sigma * taudual * dx(element_vb=BBND)
    agoal += sigma * taudual * dx(element_vb=BND)
    # agoal += sigma * taudual * dx # not needed b.c static condensation
    agoal.Assemble()

    freedofs = V_GOAL.FreeDofs() # dont care about dirichlet
    free_list = lambda L : [x for x in L if freedofs[x]]
    eblocks = [ x for x in (free_list(V_GOAL.GetDofNrs(e))
                            for e in V_GOAL.mesh.edges) if len(x) ]
    fblocks = [ x for x in (free_list(V_GOAL.GetDofNrs(f)) for f in V_GOAL.mesh.faces) if len(x) ]
    
    solve_goal = FastInv(agoal.mat, eblocks, fblocks)

    amix_mat = amix.mat if mpi_world.size == 1 else amix.mat.local_mat

    op = solve_goal @ amix_mat
    if mpi_world.size > 1:
        op = ParallelMatrix(op, row_pardofs = V_ORIGIN.ParallelDofs(),
                            col_pardofs = V_GOAL.ParallelDofs(), op = ParallelMatrix.C2C)
    return op


