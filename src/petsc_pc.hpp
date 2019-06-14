#ifndef FILE_NGSPETSC_PC_HPP
#define FILE_NGSPETSC_PC_HPP

namespace ngs_petsc_interface
{

  /** Can be used on NGSolve- or PETSc-side **/
  class PETScPreconditioner : public FlatPETScMatrix
  {
  public:

    PETScPreconditioner (shared_ptr<PETScBaseMatrix> mat, shared_ptr<ngs::BaseMatrix> _ngs_pc)
      : FlatPETScMatrix (_ngs_pc, mat->GetRowSubSet(), mat->GetColSubSet(), mat->GetRowMap(), mat->GetColMap()), petsc_amat(mat)
    { ; }

    ~PETScPreconditioner () { PCDestroy(&petsc_pc); }
    
    virtual PETScPC GetPETScPC () { return petsc_pc; }

    shared_ptr<PETScBaseMatrix> GetAMat() { return petsc_amat; }
    
  protected:
    shared_ptr<PETScBaseMatrix> petsc_amat; // The matrix this is a PC for
    PETScPC petsc_pc;
  };
  
  /**
     An NGSolve-Preconditioner, wrapped to PETSc
   **/
  class N2P_Precond : public PETScPreconditioner
  {
  public:

    N2P_Precond (shared_ptr<PETScBaseMatrix> mat, shared_ptr<ngs::BaseMatrix> _ngs_pc);

    static PetscErrorCode ApplyPC (PETScPC pc, PETScVec x, PETScVec y);

  protected:
  };

  /**
     A PETSc-Preconditioner, wrapped to NGSolve
   **/
  // class P2N_Precond : public PETScPreconditioner
  // {
  // public:
  //   P2N_Precond (PETScMat petsc_mat, string name, Array<string> options);
  // protected:
  // };

  // class PETScCompositePC : public PETScPreconditioner
  // {
  // public:
  //   PETScCompositePC (MPI_Comm comm);
  //   void AddPC (shared_ptr<PETScPreconditioner> componentpc);
  //   void Finalize ();
  //   void SetMode (PCCompositeType atype = PC_COMPOSITE_ADDITIVE);
  // protected:
  //   Array<shared_ptr<PETScPreconditioner> components;
  // };

  // class PETScFieldSplitPC : public PETScPreconditioner
  // {
  // public:
  //   PETScFieldSplitPC (shared_ptr<PETScBaseMatrix> amat);
  //   void AddFieldIndexed (Array<size_t> indices, shared_ptr<PETScPreconditioner> pc, string name = "");
  //   void AddFieldRange (size_t first, size_t next, shared_ptr<PETScPreconditioner> pc, string name = "");
  //   // void AddFieldStride (size_t first, size_t step, shared_ptr<PETScPreconditioner> pc, string name = "");
  //   void Finalize();
  // protected:
  // }
  
} // namespace ngs_petsc_interface

#endif
