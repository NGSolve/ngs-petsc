#ifndef FILE_NGSPETSC_PC_HPP
#define FILE_NGSPETSC_PC_HPP

namespace ngs_petsc_interface
{

  /** Anything that can be used as a PETSc-Preconditioner **/
  class PETScBasePrecond
  {
  public:
    PETScBasePrecond (shared_ptr<PETScBaseMatrix> _petsc_amat = nullptr, shared_ptr<PETScBaseMatrix> _petsc_pmat = nullptr,
		      string _name = "", FlatArray<string> _petsc_options = Array<string>());

    virtual PETScPC GetPETScPC () const { return petsc_pc; }
    virtual PETScPC& GetPETScPC () { return petsc_pc; }

    shared_ptr<PETScBaseMatrix> GetAMat () const { return petsc_amat; }
    void SetAMat (shared_ptr<PETScBaseMatrix> _petsc_amat) { petsc_amat = _petsc_amat; }

    shared_ptr<PETScBaseMatrix> GetPMat () const { return petsc_pmat; }
    void SetPMat (shared_ptr<PETScBaseMatrix> _petsc_pmat) { petsc_pmat = _petsc_pmat; }

    string GetName () const { return name; }

    virtual void Finalize ();
  protected:
    PETScPC petsc_pc;
    shared_ptr<PETScBaseMatrix> petsc_amat; // the matrix this is a PC for
    shared_ptr<PETScBaseMatrix> petsc_pmat; // the matrix this PC is built from (usually same as amat)
    string name;
  };


  /** A PETSc-Preconditioner **/
  // class PETScPrecond : public PETScBasePrecond
  // {
  // public:
  //   PETScPrecond (shared_ptr<PETScBaseMatrix> _petsc_amat, shared_ptr<PETScBaseMatrix> _petsc_pmat, Array<string> options, string _name = "");
  // };


  /** An NGSolve-BaseMatrix, wrapped to PETSc as a PC **/
  class NGs2PETScPrecond : public PETScBasePrecond,
			   public FlatPETScMatrix
  {
  public:
    NGs2PETScPrecond (shared_ptr<PETScBaseMatrix> _mat, shared_ptr<ngs::BaseMatrix> _ngs_pc,
		      string name = "", FlatArray<string> _petsc_options = Array<string>(), bool _finalize = true);

    ~NGs2PETScPrecond () { PCDestroy(&GetPETScPC()); }
    
    static PetscErrorCode ApplyPC (PETScPC pc, PETScVec x, PETScVec y);
  };


  class PETScCompositePC : public PETScBasePrecond
  {
  public:
    PETScCompositePC (shared_ptr<PETScBaseMatrix> _petsc_amat = nullptr, shared_ptr<PETScBaseMatrix> _petsc_pmat = nullptr,
		      string _name = "", FlatArray<string> _petsc_options = Array<string>());
    void AddPC (shared_ptr<NGs2PETScPrecond> component);
  protected:
    Array<shared_ptr<NGs2PETScPrecond>> keep_alive;
  };


  class FSField
  {
  public:
    FSField (shared_ptr<PETScBasePrecond> _pc); // 
    IS GetIS () const { return is; }
    shared_ptr<PETScBasePrecond> GetPC () const { return pc; }
    string GetName () const { return name; }
  protected:
    PETScIS is;
    shared_ptr<PETScBasePrecond> pc;
    string name;
  };


  class FSFieldRange : public FSField
  {
  public:
    FSFieldRange (shared_ptr<PETScBaseMatrix> _mat, size_t _first, size_t _next);
    FSFieldRange (shared_ptr<PETScBasePrecond> _pc, size_t _first, size_t _next);
  protected:
    void SetUpIS (shared_ptr<PETScBaseMatrix> mat, size_t _first, size_t _next);
  };


  class PETScFieldSplitPC : public PETScBasePrecond
  {
  public:
    PETScFieldSplitPC (shared_ptr<PETScBaseMatrix> amat,
		       string name = "", FlatArray<string> petsc_options = Array<string>());
    void AddField (shared_ptr<FSField> field);
    virtual void Finalize () override;
  protected:
    Array<shared_ptr<FSField>> fields;
  };
  
} // namespace ngs_petsc_interface

#endif
