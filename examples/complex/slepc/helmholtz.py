from ngsolve import *
import netgen.geom2d as g2d
import ngs_petsc_complex as petsc

geo = g2d.unit_square

ngmesh = geo.GenerateMesh(maxh=0.1)
# ngmesh.Save("scattering.vol")
mesh = Mesh(ngmesh)
# mesh = Mesh ("scattering.vol")

k = 100

V = H1(mesh, complex=True, order=3, dirichlet=".*")
u,v = V.TnT()

print("V.ndof =", V.ndof)

a = BilinearForm (V, symmetric=False)
a += grad(u) * Conj(grad(v)) * dx
a += -k*k * u * Conj(v) * dx

mass = BilinearForm (V, symmetric=False)
mass += u * v * dx


a.Assemble()
mass.Assemble()

pcma = petsc.PETScMatrix(a.mat, freedofs=V.FreeDofs())
pcmm = petsc.PETScMatrix(mass.mat, freedofs=V.FreeDofs())


eps_opts = { "eps_non_hermitian" : "",
             "eps_target_all" : "",
             "eps_type" : "arnoldi",
             # "eps_type" : "lanczos",
             # "eps_type" : "krylovschur",
             "eps_tol" : 1e-12,
             "eps_nev" : 100,          # nr. of eigenvalues to compute
             "eps_ncv" : 200,          # nr. of vecs to be used by solution algorithm (recommended to be >= 2 * nev in SLEPc manual)
             #"eps_monitor_lg" : "",
             #"eps_monitor_lg_all" : "",
             "eps_monitor" : "" }
eps = petsc.EPS(A = pcma, B = None, name="myfirsteps", slepc_options = eps_opts, finalize = True)
eps.Solve()

print("Did EPS converge? :", eps.converged)
print("Why did EPS (not) converge? :", eps.converged_reason)
print("# of converged eigen-values:", eps.nconv)
print("# of converged eigen-values:", eps.nconv)

evals = eps.evals

# print(len(evals), "evals:", evals)

# epairs = eps.GetEigenPair(19)
# print("epairs", epairs)


import matplotlib.pyplot as plt
plt.scatter([x.real for x in evals], [x.imag for x in evals], label="eigenvalues")
plt.legend()
plt.show()
