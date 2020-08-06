/** Compile with:
       ngscxx -c -I<PATH_TO_PETSC_INCLUDE> -I<PATH_TO_NGS_PETSC_INTERFACE_INCLUDED> poisson.cpp
       ngsld poisson.o -lngcomp -lngla -lngfem -lngstd -lngcore -L<PATH_TO_PETSC_LIBS> -lpetsc -L<PATH_TO_PETSC_INTERFACE_LIB> -lpetscinterface -lmpi -o poisson
**/

/** We have to be careful with namespaces - names such as "Vec" and "Mat" exist both in NGSolve and PETSc **/

#include <comp.hpp>
namespace ngs = ngcomp;

#include <petsc_interface.hpp>
namespace pci = ngs_petsc_interface;

#include <petsc.h>

int main(int argc, char** argv)
{
  /** Initializes MPI, finalizes in destructor **/
  ngs::MyMPI mpi_init(argc, argv);

  /** Initialize PETSc **/
  PetscInitialize(&argc, &argv, NULL, NULL);

  /** A wrapper around an MPI communicator **/
  ngs::NgMPI_Comm comm(MPI_COMM_WORLD);

  auto ma = make_shared<ngs::MeshAccess>("square.vol", comm); 
  ngs::Flags flags_fes;
  flags_fes.SetFlag ("order", 4);
  auto fes = make_shared<ngs::H1HighOrderFESpace> (ma, flags_fes);

  ngs::Flags flags_gfu;
  auto gfu = make_shared<ngs::T_GridFunction<double>> (fes, "u", flags_gfu);

  ngs::Flags flags_bfa;
  auto bfa = make_shared<ngs::T_BilinearFormSymmetric<double>> (fes, "a", flags_bfa);

  auto u = fes->GetTrialFunction();
  auto v = fes->GetTestFunction();

  bfa -> AddIntegrator(make_shared<ngs::SymbolicBilinearFormIntegrator>(u->Deriv() * v->Deriv(), ngs::VOL, ngs::VOL));

  ngs::Array<double> penalty(ma->GetNBoundaries());
  penalty = 0.0;
  penalty[0] = 1e10;

  auto pen = make_shared<ngs::DomainConstantCoefficientFunction> (penalty);
  bfa -> AddIntegrator(make_shared<ngs::SymbolicBilinearFormIntegrator>(pen*u*v, ngs::BND, ngs::VOL));

  ngs::Flags flags_lff;
  auto lff = make_shared<ngs::T_LinearForm<double>> (fes, "f", flags_lff);

  auto lfi = make_shared<ngs::SourceIntegrator<2>> (make_shared<ngs::ConstantCoefficientFunction> (5));
  lff -> AddIntegrator (lfi);

  ngs::LocalHeap lh(20 * 1024 * 1024);
  fes -> Update();
  fes -> FinalizeUpdate();

  gfu -> Update();
  bfa -> Assemble(lh);
  lff -> Assemble(lh);

  /** Convert to a PETSc Mat.
      The last parameter is the PETSc MatType we want the PETSc Mat to have.
      In parallel, first, a MATIS PETSc Mat is created and then converted if necessary.
      Per default it is converted to MATMPIAIJ or MATMPIBAIJ. Here, we leave it as a MATIS.
  **/
  auto mata = make_shared<pci::PETScMatrix>(bfa->GetMatrixPtr(), fes->GetFreeDofs(), fes->GetFreeDofs(),
					    (comm.Size() > 1) ? pci::PETScMatrix::MAT_TYPE::IS_AIJ : pci::PETScMatrix::MAT_TYPE::AIJ );

  if (comm.Size() > 1) {
    /** We manually convert the matrix to MPIAIJ by directly using the wrapped PETSc Mat object.  **/
    Mat petsc_mat = mata->GetPETScMat();
    MatConvert(petsc_mat, MATMPIAIJ, MAT_INPLACE_MATRIX, &petsc_mat);
  }

  /** Create a wrapper around a PETSc KSP object.
      The second parameter is a list of options for the PETSc option database.
      They have to be given in the format
         "opt<space>val" or "opt<space>"
      The last parameter is the name the KSP object will have within PETSc. Given options are prefixed accordingly.
  **/
  auto ksp = make_shared<pci::PETScKSP>(mata, ngs::Array<string>({ "ksp_type cg", "pc_type gamg", "ksp_monitor " }), "my_poisson_ksp");

  /** We can directly access the wrapped PETSc KSP object. **/
  KSPSetTolerances(ksp->GetKSP(), 1e-6, 0, 1e10, 50);

  /** Calls KSPSetUp. Prior to this point, we could for example set an NGSovle preconditioner, wrapped to PETSc
      as PCSHELL that should be used. **/
  ksp->Finalize();

  /** Solve the equation **/
  ngs::BaseVector & ng_rhs = lff -> GetVector();
  ngs::BaseVector & vecu = gfu -> GetVector();

  bool byhand = true;
  if (!byhand) {
    /** We can use the wrapped ksp as an NGSolve BaseMatrix **/
    vecu = (*ksp) * ng_rhs;
  }
  else { /** Or we manually work with the wrapped PETSc KSP object. **/
    /** This can map row-vectors between PETSc and NGSovle**/
    auto vec_map = mata->GetRowMap();
    /** Create comaptible PETSc vectors to hold the right hand side and solution vectors **/
    Vec pc_rhs = vec_map->CreatePETScVector();
    Vec pc_sol = vec_map->CreatePETScVector();
    /** Convert rhs, solve equation and convert solution back **/
    vec_map->NGs2PETSc(ng_rhs, pc_rhs);
    KSPSolve(ksp->GetKSP(), pc_rhs, pc_sol);
    vec_map->PETSc2NGs(vecu, pc_sol);
  }

  /** Finalize PETSc **/
  PetscFinalize();

  // cout << "Solution vector = " << endl << vecu << endl;
  return 0;
}
