#ifndef FILE_NGSPETSC_SNES_HPP
#define FILE_NGSPETSC_SNES_HPP

namespace ngs_petsc_interface
{

  /**
     Solves F(u) = 0
   **/
  class PETScSNES
  {
  public:

    enum JACOBI_MAT_MODE : char { APPLY = 0,       // Do not assemble jacobi mat (use ApplyLinearization)
				  FLAT = 1,        // Assemble Jacobi matrix, but only wrap it to PETSc
				  CONVERT = 2 };   // Assemble Jacobi matrix, and convert it to a PETSc matrix

    PETScSNES (shared_ptr<ngs::BilinearForm> _blf, FlatArray<string> _opts, string _name = "",
	       shared_ptr<ngs::LocalHeap> _lh = nullptr, JACOBI_MAT_MODE _jac_mode = FLAT);

    ~PETScSNES ();

    void Finalize ();
    
    INLINE SNES& GetSNES () { return snes; }
    INLINE SNES GetSNES () const { return snes; }

    INLINE shared_ptr<PETScKSP> GetKSP () const { return ksp; }

    void Solve (ngs::BaseVector & sol);

    // f = F(x)
    static PetscErrorCode EvaluateF (SNES snes, PETScVec x, PETScVec f, void* ctx);

    // A = F'(x), B is the matrix used to build the PC used for the linear solve with A
    static PetscErrorCode EvaluateJac (SNES snes, PETScVec x, PETScMat A, PETScMat B, void* ctx);

  protected:
    shared_ptr<ngs::BilinearForm> blf;
    shared_ptr<LocalHeap> use_lh;
    JACOBI_MAT_MODE mode;
    SNES snes;
    shared_ptr<PETScKSP> ksp;
    PETScVec func_vec, sol_vec;
    shared_ptr<PETScBaseMatrix> jac_mat;
    shared_ptr<ngs::BaseVector> row_vec, col_vec, lin_vec;
  };

} // namespace ngs_petsc_interface

#endif
