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

    /** makes the PETScPC ready to use **/
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

    PETSc2NGsPrecond (shared_ptr<ngs::BilinearForm> _bfa, const ngs::Flags & _aflags,
		      const string _aname = "petsc_precond");

    // does not do anything, but we need to have it in oder to register the Preconditioner
    PETSc2NGsPrecond (const ngs::PDE & _apde, const ngs::Flags & _aflags, const string _aname = "petsc_precond");

    PETSc2NGsPrecond (shared_ptr<PETScBaseMatrix> _petsc_amat = nullptr, shared_ptr<PETScBaseMatrix> _petsc_pmat = nullptr,
		      string _name = "", FlatArray<string> _petsc_options = Array<string>())
      : PETScBasePrecond(_petsc_amat, _petsc_pmat, _name, _petsc_options),
	ngs::Preconditioner( shared_ptr<ngs::BilinearForm>(), ngs::Flags({"not_register_for_auto_update"}), _name)
    { PETScBasePrecond::Finalize(); }

    virtual void Mult (const ngs::BaseVector & x, ngs::BaseVector & y) const override;
    virtual void MultAdd (double scal, const ngs::BaseVector & x, ngs::BaseVector & y) const override;
    virtual ngs::AutoVector CreateRowVector () const override { return GetAMat()->GetRowMap()->CreateNGsVector(); }
    virtual ngs::AutoVector CreateColVector () const override { return GetAMat()->GetColMap()->CreateNGsVector(); }
    virtual int VHeight () const override { return GetAMat()->GetNGsMat()->VHeight(); }
    virtual int VWidth () const override { return GetAMat()->GetNGsMat()->VWidth(); }


    virtual const BaseMatrix & GetAMatrix () const override { return *GetAMat()->GetNGsMat(); }
    virtual void InitLevel (shared_ptr<ngs::BitArray> freedofs = nullptr) override;
    virtual void FinalizeLevel (const ngs::BaseMatrix * mat = nullptr) override;
    virtual void Update ()  override { ; }
  protected:
    shared_ptr<ngs::BilinearForm> bfa;
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

#ifdef PETSC_HAVE_HYPRE

  class PETScHypreAuxiliarySpacePC : public PETSc2NGsPrecond
  {
  public:
    PETScHypreAuxiliarySpacePC (shared_ptr<ngs::BilinearForm> _bfa, const ngs::Flags & _aflags,
				const string _aname = "petsc_hypre_precond");

    PETScHypreAuxiliarySpacePC (const ngs::PDE & _apde, const ngs::Flags & _aflags, const string _aname = "petsc_hypre_precond");

    PETScHypreAuxiliarySpacePC (shared_ptr<PETScBaseMatrix> _petsc_amat = nullptr, shared_ptr<PETScBaseMatrix> _petsc_pmat = nullptr,
				string _name = "", FlatArray<string> _petsc_options = Array<string>());

    void SetGradientMatrix (shared_ptr<PETScMatrix> _grad_mat) { grad_mat = _grad_mat; }
    // void SetGradientMatrix (shared_ptr<BaseMatrix> _grad_mat);

    void SetCurlMatrix (shared_ptr<PETScMatrix> _curl_mat) { curl_mat = _curl_mat; }
    // void SetCurlMatrix (shared_ptr<BaseMatrix> _curl_mat);

    void SetHCurlEmbeddingMatrix (shared_ptr<PETScMatrix> _HC_embed) { HC_embed = _HC_embed; }
    // void SetHCurlEmbeddingMatrix (shared_ptr<BaseMatrix> _HC_embed);

    void SetHDivEmbeddingMatrix (shared_ptr<PETScMatrix> _HD_embed) { HD_embed = _HD_embed; }
    // void SetHDivEmbeddingMatrix (shared_ptr<BaseMatrix> _HD_embed);

    void SetVectorLaplaceMatrix (shared_ptr<PETScMatrix> _alpha_mat) { alpha_mat = _alpha_mat; }
    // void SetVectorLaplaceMatrix (shared_ptr<BaseMatrix> _alpha_mat);

    void SetScalarLaplaceMatrix (shared_ptr<PETScMatrix> _beta_mat) { beta_mat = _beta_mat; }
    // void SetScalarLaplaceMatrix (shared_ptr<BaseMatrix> _beta_mat);

    void SetConstantVectors (shared_ptr<ngs::BaseVector> _ozz, shared_ptr<ngs::BaseVector> _zoz, shared_ptr<ngs::BaseVector> _zzo);

    virtual void FinalizeLevel (const ngs::BaseMatrix * mat = nullptr) override;

  protected:
    PetscInt dimension = 3;                // dimension (used for AMS)
    shared_ptr<PETScMatrix> grad_mat;      // (scalar) H1 -> HC gradient matrix
    shared_ptr<PETScMatrix> curl_mat;      // HC -> HD curl
    shared_ptr<PETScMatrix> HC_embed;      // (vector) H1 -> HC embedding matrix
    shared_ptr<PETScMatrix> HD_embed;      // (vector) H1 -> HD embedding matrix
    shared_ptr<PETScMatrix> alpha_mat;     // vector stiffness matrix
    shared_ptr<PETScMatrix> beta_mat;      // scalar stiffness matrix
    shared_ptr<ngs::BaseVector> ozz, zoz, zzo;  // (1,0,0), (0,1,0) and (0,0,1) in HC basis
  };


  class PETScHypreAMS : public PETScHypreAuxiliarySpacePC
  {
  public:
    PETScHypreAMS (shared_ptr<ngs::BilinearForm> _bfa, const ngs::Flags & _aflags,
		   const string _aname = "petsc_hypre_ams_precond");

    PETScHypreAMS (const ngs::PDE & _apde, const ngs::Flags & _aflags, const string _aname = "petsc_hypre_ams_precond");

    PETScHypreAMS (shared_ptr<PETScBaseMatrix> _petsc_amat = nullptr, shared_ptr<PETScBaseMatrix> _petsc_pmat = nullptr,
		   string _name = "", FlatArray<string> _petsc_options = Array<string>());

    virtual void InitLevel (shared_ptr<ngs::BitArray> freedofs = nullptr) override;

    virtual void FinalizeLevel (const ngs::BaseMatrix * mat = nullptr) override;
  };


  // class PETScHypreADS : public PETScHypreAuxiliarySpacePC
  // {

  // };
#endif

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
