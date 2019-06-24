#ifndef FILE_NGSPETSC_KSP_HPP
#define FILE_NGSPETSC_KSP_HPP

namespace ngs_petsc_interface
{

  /**
     
   **/
  class PETScKSP : public ngs::BaseMatrix
  {
  public:

    // Wrap NGSolve-Matrix to PETSc, create a new KSP
    PETScKSP (shared_ptr<PETScBaseMatrix> _petsc_mat, FlatArray<string> _opts, string _name = "");
    PETScKSP (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _freedofs, FlatArray<string> _opts, string _name = "");

    // We already have a KSP on PETSc-Side
    PETScKSP (shared_ptr<PETScBaseMatrix> _petsc_mat, KSP _ksp);

    ~PETScKSP ();

    void SetPC (shared_ptr<PETScBasePrecond> apc);

    void Finalize ();

    shared_ptr<PETScBaseMatrix> GetMatrix () const { return petsc_mat; }
    INLINE KSP& GetKSP () { return ksp; }
    INLINE KSP GetKSP () const { return ksp; }
    
    virtual void Mult (const ngs::BaseVector & x, ngs::BaseVector & y) const override;

    virtual ngs::AutoVector CreateRowVector () const override { return GetMatrix()->GetRowMap()->CreateNGsVector(); }
    virtual ngs::AutoVector CreateColVector () const override { return GetMatrix()->GetColMap()->CreateNGsVector(); }

  protected:
    shared_ptr<PETScBaseMatrix> petsc_mat;
    shared_ptr<PETScBasePrecond> petsc_pc;
    PETScVec petsc_rhs, petsc_sol;
    KSP ksp; bool own_ksp;
  };

} // namespace ngs_petsc_interface

#endif
