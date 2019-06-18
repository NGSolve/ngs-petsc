#ifndef FILE_NGSPETSC_LINALG_HPP
#define FILE_NGSPETSC_LINALG_HPP

/**
   Matrices / Vectors
**/

namespace ngs_petsc_interface
{
  /** 
      Can convert between NGSolve- and PETSc vectors
      If _build_maps == true, stores the DOF-mapping explicitely
  **/
  class NGs2PETScVecMap
  {
  public:

    /** we have to give ndof and bs explicitely, because pardofs and subset can be nullptrs (not parallel / no Dirichlet BC)**/
    NGs2PETScVecMap (size_t _ndof, int _bs, shared_ptr<ngs::ParallelDofs> _pardofs,
		     shared_ptr<ngs::BitArray> _subset);
    
    ~NGs2PETScVecMap ();

    int GetBS () const { return bs; }
    INLINE bool IsParallel () const { return pardofs != nullptr; }
    shared_ptr<ngs::ParallelDofs> GetParallelDofs () const { return pardofs; }
    shared_ptr<ngs::BitArray> GetSubSet () const { return subset; }

    void NGs2PETSc (ngs::BaseVector& ngs_vec, PETScVec petsc_vec);
    void PETSc2NGs (ngs::BaseVector& ngs_vec, PETScVec petsc_vec);

    size_t GetNRowsLocal  () const { return nrows_loc; }
    size_t GetNRowsGlobal () const { return nrows_glob; }

    FlatArray<PetscInt> GetDOFMap () const { return dof_map; }
    ISLocalToGlobalMapping GetISMap () const;

    PETScVec CreatePETScVector () const;
    shared_ptr<ngs::BaseVector> CreateNGsVector () const;

  protected:
    size_t ndof;
    int bs;
    shared_ptr<ngs::ParallelDofs> pardofs;
    shared_ptr<ngs::BitArray> subset;
    size_t nrows_loc, nrows_glob;
    Array<PetscInt> dof_map;         // maps ALL DOFS (not rows!) to global nums, non-subset get -1
    ISLocalToGlobalMapping is_map;   // maps MASTER + SUBSET DOFS (not rows!) to global nums (only constructed if parallel)
  };

  /** Ports an NGSolve-BaseMatrix to PETSc **/
  class PETScBaseMatrix : public ngs::BaseMatrix
  {
  public:

    PETScBaseMatrix (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _row_subset, shared_ptr<ngs::BitArray> _col_subset,
		     shared_ptr<NGs2PETScVecMap> _row_map = nullptr, shared_ptr<NGs2PETScVecMap> _col_map = nullptr)
      : row_map(_row_map), col_map(_col_map), ngs_mat(_ngs_mat), row_subset(_row_subset), col_subset(_col_subset)
    { ; }

    ~PETScBaseMatrix () { MatDestroy(&petsc_mat); }

    /** Call this if the NGSolve-Matrix has changed and you want to get the new values to PETSc **/
    virtual void UpdateValues () { ; }

    // Maps that can convert the row/column space between NGSolve and PETSc
    virtual shared_ptr<NGs2PETScVecMap> GetRowMap () const { return row_map; }
    virtual shared_ptr<NGs2PETScVecMap> GetColMap () const { return col_map; }

    // The PETSc-Matrix 
    virtual PETScMat GetPETScMat () const { return petsc_mat; }

    // The underlying NGSolve-Matrix, and the subsets that define the sub-block
    // of the PETSc-Matrix
    virtual INLINE shared_ptr<ngs::BaseMatrix> GetNGsMat () const { return ngs_mat; }
    virtual INLINE shared_ptr<ngs::BitArray> GetRowSubSet () const { return row_subset; }
    virtual INLINE shared_ptr<ngs::BitArray> GetColSubSet () const { return col_subset; }

    void SetNullSpace (MatNullSpace null_space);
    void SetNearNullSpace (MatNullSpace null_space);

    virtual int VHeight () const override { return GetNGsMat()->VHeight(); }
    virtual int VWidth () const override { return GetNGsMat()->VWidth(); }
    virtual ngs::AutoVector CreateRowVector () const override { return GetNGsMat()->CreateRowVector(); }
    virtual ngs::AutoVector CreateColVector () const override { return GetNGsMat()->CreateColVector(); }
    virtual void Mult (const ngs::BaseVector & x, ngs::BaseVector & y) const override
    { GetNGsMat()->Mult(x, y); }
    virtual void MultAdd (double scal, const ngs::BaseVector & x, ngs::BaseVector & y) const override
    { GetNGsMat()->MultAdd(scal, x, y); }
    virtual void MultTransAdd (double scal, const ngs::BaseVector & x, ngs::BaseVector & y) const override
    { GetNGsMat()->MultTransAdd(scal, x, y); }
    
  protected:
    shared_ptr<NGs2PETScVecMap> row_map, col_map;
    shared_ptr<ngs::BaseMatrix> ngs_mat;
    shared_ptr<ngs::BitArray> row_subset, col_subset;
    PETScMat petsc_mat;
  };


  /**
     Convert an NGSolve-Matrix to a PETSc Matrix
     linalg takes place on PETSc-side 

     Needs a ParallelMatrix with some SparseMatrix inside.
  **/
  class PETScMatrix : public PETScBaseMatrix
  {
  public:
    PETScMatrix (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _row_subset,
		 shared_ptr<ngs::BitArray> _col_subset, shared_ptr<NGs2PETScVecMap> _row_map = nullptr,
		 shared_ptr<NGs2PETScVecMap> _col_map = nullptr);

    PETScMatrix (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _row_subset,
		 shared_ptr<ngs::BitArray> _col_subset, PETScMatType _petsc_mat_type,
		 shared_ptr<NGs2PETScVecMap> _row_map = nullptr, shared_ptr<NGs2PETScVecMap> _col_map = nullptr);

    virtual void UpdateValues ();
  };


  /**
     Wrapper around a NGSolve-Matrix
     linalg takes place on NGSolve-side
      
     Can be any kind of BaseMatrix
  **/
  class FlatPETScMatrix : public PETScBaseMatrix
  {
  public:

    FlatPETScMatrix (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _row_subset,
		     shared_ptr<ngs::BitArray> _col_subset, shared_ptr<NGs2PETScVecMap> _row_map = nullptr,
		     shared_ptr<NGs2PETScVecMap> _col_map = nullptr);

  protected:
    static PetscErrorCode MatMult (PETScMat A, PETScVec x, PETScVec y);
    shared_ptr<ngs::BaseVector> row_hvec, col_hvec;
  };


  MatNullSpace NullSpaceCreate (FlatArray<shared_ptr<ngs::BaseVector>> vecs, shared_ptr<NGs2PETScVecMap> map,
				bool is_orthonormal = false, bool const_kernel = false);
  
} // namespace ngs_petsc_interface

#endif
