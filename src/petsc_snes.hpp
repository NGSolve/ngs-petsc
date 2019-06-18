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

    PETScSNES (shared_ptr<ngs::BilinearForm> _blf, FlatArray<string> _opts, string _name = "",
	       shared_ptr<ngs::LocalHeap> alh = nullptr);

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
    SNES snes;
    shared_ptr<PETScKSP> ksp;
    PETScVec func_vec, sol_vec;
    shared_ptr<PETScBaseMatrix> jac_mat;
    shared_ptr<ngs::BaseVector> row_vec, col_vec;
  };

} // namespace ngs_petsc_interface

#endif
