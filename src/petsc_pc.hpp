#ifndef FILE_NGSPETSC_PC_HPP
#define FILE_NGSPETSC_PC_HPP

namespace ngs_petsc_interface
{

  /** Anything that can be used as a PETSc-Preconditioner **/
  class PETScBasePrecond
  {
  public:
    PETScBasePrecond (MPI_Comm comm, string _name = "", FlatArray<string> _petsc_options = Array<string>());

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
    PETScVec petsc_rhs, petsc_sol;
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

    ~NGs2PETScPrecond () { /* PCDestroy(&GetPETScPC()); */ }
    
    static PetscErrorCode ApplyPC (PETScPC pc, PETScVec x, PETScVec y);
  };


  class PETSc2NGsPrecond : public PETScBasePrecond,
			   public ngs::Preconditioner
  {
  public:

    PETSc2NGsPrecond (shared_ptr<ngs::BilinearForm> bfa, const ngs::Flags & aflags,
		      const string aname = "petsc_precond");

    // does not do anything, but we need to have it in oder to register the Preconditioner
    PETSc2NGsPrecond (const ngs::PDE & apde, const ngs::Flags & aflags, const string aname = "precond");

    PETSc2NGsPrecond (shared_ptr<PETScBaseMatrix> _petsc_amat = nullptr, shared_ptr<PETScBaseMatrix> _petsc_pmat = nullptr,
		      string _name = "", FlatArray<string> _petsc_options = Array<string>())
      : PETScBasePrecond(_petsc_amat, _petsc_pmat, _name, _petsc_options),
	ngs::Preconditioner( shared_ptr<ngs::BilinearForm>(), ngs::Flags({"not_register_for_auto_update"}), _name)
    { PETScBasePrecond::Finalize(); }

    virtual void Mult (const ngs::BaseVector & x, ngs::BaseVector & y) const override;
    virtual ngs::AutoVector CreateRowVector () const override { return GetAMat()->GetRowMap()->CreateNGsVector(); }
    virtual ngs::AutoVector CreateColVector () const override { return GetAMat()->GetColMap()->CreateNGsVector(); }
    virtual int VHeight () const override { return GetAMat()->GetNGsMat()->VHeight(); }
    virtual int VWidth () const override { return GetAMat()->GetNGsMat()->VWidth(); }


    virtual void InitLevel (shared_ptr<ngs::BitArray> freedofs = nullptr) override;
    virtual void FinalizeLevel (const ngs::BaseMatrix * mat = nullptr) override;
    virtual void Update ()  override { ; }
  protected:
    using PETScBasePrecond::name; // there is also a name in Preconditioner
    shared_ptr<BitArray> subset; // only used to stash freedofs given in InitLevel
  };


  class PETScCompositePC : public PETSc2NGsPrecond
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
    FSField (shared_ptr<PETScBasePrecond> _pc, string name = ""); // 
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
    FSFieldRange (shared_ptr<PETScBaseMatrix> _mat, size_t _first, size_t _next, string name = "");
    FSFieldRange (shared_ptr<PETScBasePrecond> _pc, size_t _first, size_t _next, string name = "");
  protected:
    void SetUpIS (shared_ptr<PETScBaseMatrix> mat, size_t _first, size_t _next);
  };


  class PETScFieldSplitPC : public PETSc2NGsPrecond
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
