#include "petsc_interface.hpp"

#include <python_ngstd.hpp> 

namespace ngs_petsc_interface
{

  template<class TM> INLINE typename ngs::mat_traits<TM>::TSCAL* get_ptr (TM & val) { return &val(0,0); }
  template<> INLINE double* get_ptr<double> (double & val) { return &val; }
  template<> INLINE Complex* get_ptr<Complex> (Complex & val) { return &val; }


  template<class TM>
  void SetPETScMatSeq (PETScMat petsc_mat, shared_ptr<ngs::SparseMatrixTM<TM>> spmat,
		       shared_ptr<ngs::BitArray> rss, shared_ptr<ngs::BitArray> css)
  {
    static ngs::Timer t(string("SetPETScMatSeq<Mat<") + to_string(ngs::mat_traits<TM>::HEIGHT) + string(">>")); ngs::RegionTimer rt(t);

    PETScInt bs; MatGetBlockSize(petsc_mat, &bs);
    if (bs != ngs::mat_traits<TM>::WIDTH) {
      throw Exception(string("Block-Size of petsc-mat (") + to_string(bs) + string(") != block-size of ngs-mat(")
		      + to_string(ngs::mat_traits<TM>::WIDTH) + string(")"));
    }
	
    // row map (map for a row)
    PETScInt bw = ngs::mat_traits<TM>::WIDTH;
    int nbrow = 0;
    Array<int> row_compress(spmat->Width());
    for (auto k : Range(spmat->Width()))
      { row_compress[k] = (!rss || rss->Test(k)) ? nbrow++ : -1; }
    int ncols = nbrow * bw;
    
    // col map (map for a col)
    PETScInt bh = ngs::mat_traits<TM>::HEIGHT;
    int nbcol = 0;
    Array<int> col_compress(spmat->Height());
    for (auto k : Range(spmat->Height()))
      { col_compress[k] = (!css || css->Test(k)) ? nbcol++ : -1; }
    int nrows = nbcol * bh;

    bool symmetric = dynamic_pointer_cast<ngs::SparseMatrixSymmetric<TM>>(spmat) != nullptr;

    size_t len_vals = 0;
    for (auto k : Range(spmat->Height())) {
      PETScInt ck = col_compress[k];
      if (ck != -1) {
	auto ris = spmat->GetRowIndices(k);
	auto rvs = spmat->GetRowValues(k);
	for (auto j : Range(ris.Size())) {
	  PETScInt cj = row_compress[ris[j]];
	  if (cj != -1) {
	    PETScScalar* data = get_ptr(rvs[j]);
	    MatSetValuesBlocked(petsc_mat, 1, &ck, 1, &cj, data, INSERT_VALUES);
	    if (symmetric)
	      { MatSetValuesBlocked(petsc_mat, 1, &cj, 1, &ck, data, INSERT_VALUES); }
	  }
	}
      }
    }
    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);
  }


  template<class TM>
  void SetPETScMatIS (PETScMat petsc_mat, shared_ptr<ngs::SparseMatrixTM<TM>> spmat, shared_ptr<ngs::BitArray> rss, shared_ptr<ngs::BitArray> css)
  {
    /** If the PETSc-Mat is in MATIS format, we can just directly replace entries in it's local matrix **/

    PETScMat local_mat; MatISGetLocalMat(petsc_mat, &local_mat);
    SetPETScMatSeq (local_mat, spmat, rss, css);
  } // SetPETScMatPar


  template<class TM>
  void SetPETScMatPar (PETScMat petsc_mat, shared_ptr<ngs::SparseMatrixTM<TM>> spmat, shared_ptr<NGs2PETScVecMap> row_map, shared_ptr<NGs2PETScVecMap> col_map)
  {
    /**
       We have to zero out the matrix and then ADD values (instead of SET them)
       because in NGSolve we have an "IS"-Style matrix (overlapping diagonal blocks, with one block per proc)
       but petsc_mat is in MATMPIAIJ or MATMPIBAIJ format, which is simply distributed row-wise
     **/

    static ngs::Timer t(string("SetPETScMatPar<Mat<") + to_string(ngs::mat_traits<TM>::HEIGHT) + string(">>")); ngs::RegionTimer rt(t);

    PETScInt bs = ngs::mat_traits<TM>::WIDTH;

    auto row_dm = row_map->GetDOFMap();
    auto col_dm = col_map->GetDOFMap();
    
    MatZeroEntries(petsc_mat);

    for (auto k : Range(spmat->Height())) {
      PETScInt ck = col_dm[k];
      if (ck != -1) {
	auto ris = spmat->GetRowIndices(k);
	auto rvs = spmat->GetRowValues(k);
	for (auto j : Range(ris.Size())) {
	  PETScInt cj = row_dm[ris[j]];
	  if (cj != -1) {
	    PETScScalar* data = get_ptr(rvs[j]);
	    MatSetValuesBlocked(petsc_mat, 1, &ck, 1, &cj, data, ADD_VALUES);
	  }
	}
      }
    }
    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);
  } // SetPETScMatPar


  template<class TM>
  void SetPETScMat (PETScMat petsc_mat, shared_ptr<ngs::SparseMatrixTM<TM>> spmat, shared_ptr<NGs2PETScVecMap> row_map, shared_ptr<NGs2PETScVecMap> col_map)
  {
    MatType petsc_type; MatGetType(petsc_mat, &petsc_type); string type(petsc_type);
    if (type == string(MATIS))
      { SetPETScMatIS(petsc_mat, spmat, row_map->GetSubSet(), col_map->GetSubSet()); }
    else if ( (type == string(MATMPIAIJ)) || (type == string(MATMPIBAIJ)) )
      { SetPETScMatPar(petsc_mat, spmat, row_map, col_map); }
    else if ( (type == string(MATSEQAIJ)) || (type == string(MATSEQBAIJ)) )
      { SetPETScMatSeq(petsc_mat, spmat, row_map->GetSubSet(), col_map->GetSubSet()); }
    else
      { throw Exception("Cannot update values for PETSc matrix of this type!!"); }
  }

  template<class TM>
  PETScMat CreatePETScMatSeqBAIJFromSymmetric (shared_ptr<ngs::SparseMatrixSymmetric<TM>> spmat, shared_ptr<ngs::BitArray> rss, shared_ptr<ngs::BitArray> css)
  {
    static ngs::Timer t(string("CreatePETScMatSeqBAIJFromSymmetric<Mat<") + to_string(ngs::mat_traits<TM>::HEIGHT) + string(">>")); ngs::RegionTimer rt(t);

    // row map (map for a row)
    PETScInt bw = ngs::mat_traits<TM>::WIDTH;
    int nbrow = 0;
    Array<int> row_compress(spmat->Width());
    for (auto k : Range(spmat->Width()))
      { row_compress[k] = (!rss || rss->Test(k)) ? nbrow++ : -1; }
    int ncols = nbrow * bw;

    // col map (map for a col)
    PETScInt bh = ngs::mat_traits<TM>::HEIGHT;
    int nbcol = 0;
    Array<int> col_compress(spmat->Height());
    for (auto k : Range(spmat->Height()))
      { col_compress[k] = (!css || css->Test(k)) ? nbcol++ : -1; }
    int nrows = nbcol * bh;

    // allocate mat
    Array<PETScInt> nzepr(nbcol+1); nzepr = 0;
    for (auto k : Range(spmat->Height())) {
      auto ck = col_compress[k];
      if (ck != -1) {
	for (auto j : spmat->GetRowIndices(k)) {
	  auto cj = row_compress[j];
	  if (cj != -1)
	    { nzepr[ck+1]++; if (cj != ck) { nzepr[cj+1]++; } }
	}
      }
    }
    PETScMat petsc_mat;
    if (bh == 1)
      { MatCreateSeqAIJ(PETSC_COMM_SELF, nrows, ncols, -1, &nzepr[1], &petsc_mat); }
    else
      { MatCreateSeqBAIJ(PETSC_COMM_SELF, bh, nrows, ncols, -1, &nzepr[1], &petsc_mat); }

    // cols
    for (auto k : Range(1, nbcol+1))
      { nzepr[k] += nzepr[k-1]; }
    Array<int> cnt(nbcol); cnt = 0;
    Array<PETScInt> cols(nzepr.Last());
    for (auto k : Range(spmat->Height())) {
      auto ck = col_compress[k];
      if (ck != -1) {
	for (auto j : spmat->GetRowIndices(k)) {
	  auto cj = row_compress[j];
	  if (cj != -1) {
	    cols[nzepr[ck] + cnt[ck]++] = cj;
	    if (cj != ck)
	      { cols[nzepr[cj] + cnt[cj]++] = ck; }
	  }
	}
      }
    }
    if (bh == 1)
      { MatSeqAIJSetColumnIndices(petsc_mat, &cols[0]); }
    else
      { MatSeqBAIJSetColumnIndices(petsc_mat, &cols[0]); }

    // vals
    for (auto k : Range(spmat->Height())) {
      auto ck = col_compress[k];
      if (ck != -1) {
	auto ris = spmat->GetRowIndices(k);
	auto rvs = spmat->GetRowValues(k);
	for (auto j : Range(ris.Size())) {
	  auto cj = row_compress[ris[j]];
	  if (cj != -1) {
	    PETScScalar* data = get_ptr(rvs[j]);
	    MatSetValuesBlocked(petsc_mat, 1, &ck, 1, &cj, data, INSERT_VALUES);
	    if (cj != ck)
	      { MatSetValuesBlocked(petsc_mat, 1, &cj, 1, &ck, data, INSERT_VALUES); }
	  }
	}
      }
    }
    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);

    // cout << "SYMM SPMAT: " << endl << *spmat << endl;
    // cout << "SYMM-TO-FULL PETSC BLOCK-MAT: " << endl;
    // MatView(petsc_mat, PETSC_VIEWER_STDOUT_SELF);
    return petsc_mat;
  } // CreatePETScMatSeqBAIJFromSymmetric


  template<class TM>
  PETScMat CreatePETScMatSeqBAIJ (shared_ptr<ngs::SparseMatrixTM<TM>> spmat, shared_ptr<ngs::BitArray> rss, shared_ptr<ngs::BitArray> css)
  {

    static_assert(ngs::mat_traits<TM>::WIDTH == ngs::mat_traits<TM>::HEIGHT, "PETSc can only handle square block entries!");

    static ngs::Timer t(string("CreatePETScMatSeqBAIJ<Mat<") + to_string(ngs::mat_traits<TM>::HEIGHT) + string(">>")); ngs::RegionTimer rt(t);

    if (auto sym_spm = dynamic_pointer_cast<ngs::SparseMatrixSymmetric<TM>>(spmat))
      { return CreatePETScMatSeqBAIJFromSymmetric (sym_spm, rss, css); }

    // row map (map for a row)
    PETScInt bw = ngs::mat_traits<TM>::WIDTH;
    int nbrow = 0;
    Array<int> row_compress(spmat->Width());
    for (auto k : Range(spmat->Width()))
      { row_compress[k] = (!rss || rss->Test(k)) ? nbrow++ : -1; }
    int ncols = nbrow * bw;

    // col map (map for a col)
    PETScInt bh = ngs::mat_traits<TM>::HEIGHT;
    int nbcol = 0;
    Array<int> col_compress(spmat->Height());
    for (auto k : Range(spmat->Height()))
      { col_compress[k] = (!css || css->Test(k)) ? nbcol++ : -1; }
    int nrows = nbcol * bh;

    // allocate mat
    Array<PETScInt> nzepr(nbcol); nzepr = 0;
    nbcol = 0;
    for (auto k : Range(spmat->Height())) {
      if (!css || css->Test(k)) {
	auto & c = nzepr[nbcol++];
	for (auto j : spmat->GetRowIndices(k))
	  if (!rss || rss->Test(j))
	    { c++; }
      }
    }
    PETScMat petsc_mat;
    if (bh == 1)
      { MatCreateSeqAIJ(PETSC_COMM_SELF, nrows, ncols, -1, &nzepr[0], &petsc_mat); }
    else
      { MatCreateSeqBAIJ(PETSC_COMM_SELF, bh, nrows, ncols, -1, &nzepr[0], &petsc_mat); }

    // cols
    int n_b_entries = 0;
    for (auto k : Range(nzepr.Size()))
      { n_b_entries += nzepr[k]; }
    Array<PETScInt> cols(n_b_entries);
    n_b_entries = 0;
    for (auto k : Range(spmat->Height())) {
      if (!css || css->Test(k)) {
	for (auto j : spmat->GetRowIndices(k))
	  if (!rss || rss->Test(j))
	    { cols[n_b_entries++] = row_compress[j]; }
      }
    }
    if (bh == 1)
      { MatSeqAIJSetColumnIndices(petsc_mat, &cols[0]); }
    else
      { MatSeqBAIJSetColumnIndices(petsc_mat, &cols[0]); }

    // vals
    size_t len_vals = 0;
    for (auto k : Range(spmat->Height())) {
      PETScInt ck = col_compress[k];
      if (ck != -1) {
	auto ris = spmat->GetRowIndices(k);
	auto rvs = spmat->GetRowValues(k);
	for (auto j : Range(ris.Size())) {
	  PETScInt cj = row_compress[ris[j]];
	  if (cj != -1) {
	    PETScScalar* data = get_ptr(rvs[j]);
	    MatSetValuesBlocked(petsc_mat, 1, &ck, 1, &cj, data, INSERT_VALUES);
	  }
	}
      }
    }
    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);

    // cout << "SPMAT: " << endl << *spmat << endl;
    // cout << "PETSC BLOCK-MAT: " << endl;
    // MatView(petsc_mat, PETSC_VIEWER_STDOUT_SELF);
    return petsc_mat;
  } // CreatePETScMatSeqBAIJ


  PETScMat CreatePETScMatSeq (shared_ptr<ngs::BaseMatrix> mat, shared_ptr<ngs::BitArray> rss, shared_ptr<ngs::BitArray> css)
  {
    PETScMat ret = NULL;
    Iterate<MAX_SYS_DIM>([&](auto n) {
	constexpr int N = 1 + n;
	if constexpr(N==1) {
	    if (auto spm = dynamic_pointer_cast<ngs::SparseMatrixTM<PETScScalar>>(mat))
	      { ret = CreatePETScMatSeqBAIJ(spm, rss, css); }
	  }
	else {
	  if (auto spm = dynamic_pointer_cast<ngs::SparseMatrixTM<ngs::Mat<N, N, PETScScalar>>>(mat))
	    { ret = CreatePETScMatSeqBAIJ(spm, rss, css); }
	}
      });
    return ret;
  } // CreatePETScMatSeq


  /**
     Only leaves entries in the matrix where we are master of row-dof OR col-dof.
     ( -> do this for the local mats of a C2C-ParallelMatrix)
   **/
  template<class TM>
  void DeleteDuplicateValuesTM (PETScMat petsc_mat, shared_ptr<ngs::SparseMatrixTM<TM>> spmat,
			      shared_ptr<ngs::ParallelDofs> pdrow, shared_ptr<ngs::ParallelDofs> pdcol,
			      shared_ptr<ngs::BitArray> rss, shared_ptr<ngs::BitArray> css)
  {

    static ngs::Timer t(string("DeleteDuplicateValuesTM<Mat<") + to_string(ngs::mat_traits<TM>::HEIGHT) + string(">>")); ngs::RegionTimer rt(t);

    PETScInt bs; MatGetBlockSize(petsc_mat, &bs);
    if (bs != ngs::mat_traits<TM>::WIDTH) {
      throw Exception(string("Block-Size of petsc-mat (") + to_string(bs) + string(") != block-size of ngs-mat(")
		      + to_string(ngs::mat_traits<TM>::WIDTH) + string(")"));
    }
	
    // row map (map for a row)
    PETScInt bw = ngs::mat_traits<TM>::WIDTH;
    int nbrow = 0;
    Array<int> row_compress(spmat->Width());
    for (auto k : Range(spmat->Width()))
      { row_compress[k] = (!rss || rss->Test(k)) ? nbrow++ : -1; }
    int ncols = nbrow * bw;
    
    // col map (map for a col)
    PETScInt bh = ngs::mat_traits<TM>::HEIGHT;
    int nbcol = 0;
    Array<int> col_compress(spmat->Height());
    for (auto k : Range(spmat->Height()))
      { col_compress[k] = (!css || css->Test(k)) ? nbcol++ : -1; }
    int nrows = nbcol * bh;
    
    TM zero(0);
    PETScScalar* data = get_ptr(zero);

    for (auto k : Range(spmat->Height())) {
      auto ck = col_compress[k];
      if ( (ck == -1) || pdcol->IsMasterDof(k) ) continue;
      auto ri = spmat->GetRowIndices(k);
      for (auto j : Range(ri.Size())) {
	auto cj = row_compress[ri[j]];
	if ( (cj != -1) && !pdrow->IsMasterDof(ri[j]) )
	  { MatSetValuesBlocked(petsc_mat, 1, &ck, 1, &cj, data, INSERT_VALUES); }
      }
    }

    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);
  }


  void DeleteDuplicateValues (PETScMat petsc_mat, shared_ptr<ngs::BaseMatrix> mat,
			      shared_ptr<ngs::ParallelDofs> pdrow, shared_ptr<ngs::ParallelDofs> pdcol,
			      shared_ptr<ngs::BitArray> rss, shared_ptr<ngs::BitArray> css)
  {
    Iterate<MAX_SYS_DIM>([&](auto n) {
	constexpr int N = 1 + n;
	if constexpr(N==1) {
	    if (auto spmat = dynamic_pointer_cast<ngs::SparseMatrixTM<PETScScalar>>(mat))
	      { DeleteDuplicateValuesTM (petsc_mat, spmat, pdrow, pdcol, rss, css) ; }
	  }
	else {
	  if (auto spmat = dynamic_pointer_cast<ngs::SparseMatrixTM<ngs::Mat<N, N, PETScScalar>>>(mat))
	    { DeleteDuplicateValuesTM (petsc_mat, spmat, pdrow, pdcol, rss, css) ; }
	}
      });
  }


  void PETScBaseMatrix :: SetNullSpace (MatNullSpace null_space)
  {
    MatSetNullSpace(GetPETScMat(), null_space);
  }


  void PETScBaseMatrix :: SetNearNullSpace (MatNullSpace null_space)
  {
    MatSetNearNullSpace(GetPETScMat(), null_space);
  }


  PETScMat CreatePETScMatIS (PETScMat petsc_mat_loc,
			     shared_ptr<NGs2PETScVecMap> row_map,
			     shared_ptr<NGs2PETScVecMap> col_map)
  {

    PETScMat petsc_mat;
    MatCreateIS(row_map->GetParallelDofs()->GetCommunicator(), row_map->GetBS(),
		col_map->GetNRowsLocal(), row_map->GetNRowsLocal(),
		col_map->GetNRowsGlobal(), row_map->GetNRowsGlobal(),
		col_map->GetISMap(), row_map->GetISMap(), &petsc_mat);

    MatISSetLocalMat(petsc_mat, petsc_mat_loc);
    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);

    return petsc_mat;
  } // CreatePETScMatIS


  NGs2PETScVecMap :: NGs2PETScVecMap (size_t _ndof, int _bs, shared_ptr<ngs::ParallelDofs> _pardofs,
				      shared_ptr<ngs::BitArray> _subset)
    : ndof(_ndof), bs(_bs), pardofs(_pardofs), subset(_subset)
  {
    static ngs::Timer t("NGs2PETScVecMap constructor"); ngs::RegionTimer rt(t);
    dof_map.SetSize(ndof); dof_map = -1;
    if ( (!pardofs) && (!subset) ) {
      nrows_loc = bs * ndof;
      for (auto k : Range(ndof))
	{ dof_map[k] = k; }
    }
    else {
      nrows_loc = 0;
      for (auto k : Range(ndof))
	if ( (!pardofs || pardofs->IsMasterDof(k)) && (!subset || subset->Test(k)))
	  { nrows_loc += bs; }
      Array<int> globnums;
      if (pardofs) {
	int glob_nd = 0;
	size_t cnt = 0;
	Array<PETScInt> compress_globnums(ndof);
	pardofs->EnumerateGlobally(subset, globnums, glob_nd);
	for (auto k : Range(ndof)) {
	  dof_map[k] = globnums[k]; // (PETScInt != int is possibe, but EnumerateGlobally only for ints...)
	  if (globnums[k] != -1)
	    { compress_globnums[cnt++] = globnums[k]; }
	}
	compress_globnums.SetSize(cnt);
	ISLocalToGlobalMappingCreate(pardofs->GetCommunicator(), bs, compress_globnums.Size(), &compress_globnums[0], PETSC_COPY_VALUES, &is_map);
      }
      else // subset + sequential
	{
	  int cnt = 0;
	  for (auto k : Range(ndof))
	    if (subset->Test(k))
	      { dof_map[k] = cnt++; }
	}
    }
    nrows_glob = (pardofs == nullptr) ? nrows_loc : pardofs->GetCommunicator().AllReduce(nrows_loc, MPI_SUM);
  }

  NGs2PETScVecMap :: ~NGs2PETScVecMap ()
  {
    if (IsParallel())
      { /* ISLocalToGlobalMappingDestroy(&is_map); */ }
  }


  ISLocalToGlobalMapping NGs2PETScVecMap :: GetISMap () const
  {
    if (!IsParallel())
      { throw Exception("Called NGs2PETScVecMap::GetISMap, but is not parallel, so no IS-map built!!"); }
    return is_map;
  }


  void NGs2PETScVecMap :: NGs2PETSc (ngs::BaseVector& ngs_vec, PETScVec petsc_vec)
  {
    static ngs::Timer t("NGs2PETScVecMap::NGs2PETSc"); ngs::RegionTimer rt(t);
    ngs_vec.Cumulate();
    PETScScalar * pvs; VecGetArray(petsc_vec, &pvs);
    size_t cnt = 0;
    auto fv = ngs_vec.FV<PETScScalar>();
    for (auto k : Range(ndof))
      if ( (!pardofs || pardofs->IsMasterDof(k)) && (!subset || subset->Test(k)))
	for (auto l : Range(bs))
	  { pvs[cnt++] = fv(bs*k+l); }
    VecRestoreArray(petsc_vec, &pvs);
  } // NGs2PETSc


  template<class TSCAL>
  INLINE void NGs2PETScVecMap :: AddNGs2PETSc_impl (TSCAL scal, ngs::BaseVector& ngs_vec, PETScVec petsc_vec)
  {
    static ngs::Timer t("NGs2PETScVecMap::AddNGs2PETSc"); ngs::RegionTimer rt(t);
    ngs_vec.Cumulate();
    PETScScalar * pvs; VecGetArray(petsc_vec, &pvs);
    size_t cnt = 0;
    auto fv = ngs_vec.FV<PETScScalar>();
    for (auto k : Range(ndof))
      if ( (!pardofs || pardofs->IsMasterDof(k)) && (!subset || subset->Test(k)))
	for (auto l : Range(bs))
	  { pvs[cnt++] += scal * fv(bs*k+l); }
    VecRestoreArray(petsc_vec, &pvs);
  } // NGs2PETScVecMap::AddNGs2PETSc


  void NGs2PETScVecMap :: AddNGs2PETSc (double scal, ngs::BaseVector& ngs_vec, PETScVec petsc_vec)
  {
    AddNGs2PETSc_impl<double>(scal, ngs_vec, petsc_vec);
  } // NGs2PETScVecMap :: AddNGs2PETSc


  void NGs2PETScVecMap :: AddNGs2PETSc (Complex scal, ngs::BaseVector& ngs_vec, PETScVec petsc_vec)
  {
#ifdef PETSC_INTERFACE_COMPLEX
    AddNGs2PETSc_impl<Complex>(scal, ngs_vec, petsc_vec);
#endif
  } // NGs2PETScVecMap :: AddNGs2PETSc


  void NGs2PETScVecMap :: PETSc2NGs (ngs::BaseVector& ngs_vec, PETScVec petsc_vec)
  {
    static ngs::Timer t("NGs2PETScVecMap::PETSc2NGs"); ngs::RegionTimer rt(t);
    ngs_vec.Distribute();
    const PETScScalar * pvs; VecGetArrayRead(petsc_vec, &pvs);
    size_t cnt = 0;
    auto fv = ngs_vec.FV<PETScScalar>();
    for (auto k : Range(ndof))
      if ( (!pardofs || pardofs->IsMasterDof(k)) && (!subset || subset->Test(k)))
	for (auto l : Range(bs))
	  { fv(bs*k+l) = pvs[cnt++]; }
      else
	for (auto l : Range(bs))
	  { fv(bs*k+l) = 0; }
    VecRestoreArrayRead(petsc_vec, &pvs);
  } // PETSc2NGs


  template<class TSCAL>
  INLINE void NGs2PETScVecMap :: AddPETSc2NGs_impl (TSCAL scal, ngs::BaseVector& ngs_vec, PETScVec petsc_vec)
  {
    static ngs::Timer t("NGs2PETScVecMap::AddPETSc2NGs"); ngs::RegionTimer rt(t);
    ngs_vec.Distribute();
    const PETScScalar * pvs; VecGetArrayRead(petsc_vec, &pvs);
    size_t cnt = 0;
    auto fv = ngs_vec.FV<PETScScalar>();
    for (auto k : Range(ndof))
      if ( (!pardofs || pardofs->IsMasterDof(k)) && (!subset || subset->Test(k)))
	for (auto l : Range(bs))
	  { fv(bs*k+l) += scal * pvs[cnt++]; }
    VecRestoreArrayRead(petsc_vec, &pvs);
  } // AddPETSc2NGs
  

  void NGs2PETScVecMap :: AddPETSc2NGs (double scal, ngs::BaseVector& ngs_vec, PETScVec petsc_vec)
  {
    AddPETSc2NGs_impl<double>(scal, ngs_vec, petsc_vec);
  } // NGs2PETScVecMap :: AddPETSc2NGs

  void NGs2PETScVecMap :: AddPETSc2NGs (Complex scal, ngs::BaseVector& ngs_vec, PETScVec petsc_vec)
  {
#ifdef PETSC_INTERFACE_COMPLEX
    AddPETSc2NGs_impl<Complex>(scal, ngs_vec, petsc_vec);
#endif
  } // NGs2PETScVecMap :: AddPETSc2NGs


  unique_ptr<ngs::BaseVector> NGs2PETScVecMap :: CreateNGsVector () const
  {
    if (pardofs)
      { return make_unique<ngs::S_ParallelBaseVectorPtr<PETScScalar>> (pardofs->GetNDofLocal(), pardofs->GetEntrySize(), pardofs, ngs::DISTRIBUTED); }
    else
      { return make_unique<ngs::S_BaseVectorPtr<PETScScalar>> (ndof, bs); }
  } // CreateNGsVector

  PETScVec NGs2PETScVecMap :: CreatePETScVector () const
  {
    PETScVec v;
    if (pardofs == nullptr)
      { VecCreateSeq(PETSC_COMM_SELF, nrows_loc, &v); }
    else
      { VecCreateMPI(pardofs->GetCommunicator(), nrows_loc, nrows_glob, &v); }
    return v;
  } // CreatePETScVector


  PETScMatrix :: PETScMatrix (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _row_subset,
			      shared_ptr<ngs::BitArray> _col_subset, shared_ptr<NGs2PETScVecMap> _row_map,
			      shared_ptr<NGs2PETScVecMap> _col_map)
    : PETScBaseMatrix(_ngs_mat, _row_subset, _col_subset, _row_map, _col_map)
  {
    static ngs::Timer t("PETScMatrix constructor 1"); ngs::RegionTimer rt(t);

    ConvertMat();

    /**
       No matrix format specified, so we pick a format for the parallel matrix that fits the local matrix.
         SEQAIJ - MPIAIJ
         SEQBAIJ - MPIBAIJ
       If not parallel, nothing to do.
     **/
    PETScMatType pmt, mt;
    MatGetType(petsc_mat, &pmt);
    if (string(pmt) == string(MATIS)) {
      // get the format of the local matrix
      PETScMat loc_mat;
      MatISGetLocalMat(petsc_mat, &loc_mat);
      MatGetType(loc_mat, &mt);
      MatISRestoreLocalMat(petsc_mat, &loc_mat);
      // convert the parallel matrix
      pmt = (mt == string(MATSEQAIJ)) ? MATMPIAIJ : MATMPIBAIJ;
      MatConvert(petsc_mat, pmt, MAT_INPLACE_MATRIX, &petsc_mat);
    }

  }


  PETScMatrix :: PETScMatrix (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _row_subset,
			      shared_ptr<ngs::BitArray> _col_subset, PETScMatrix::MAT_TYPE _petsc_mat_type,
			      shared_ptr<NGs2PETScVecMap> _row_map, shared_ptr<NGs2PETScVecMap> _col_map)
    : PETScBaseMatrix (_ngs_mat, _row_subset, _col_subset, _row_map, _col_map)
  {

    static ngs::Timer t("PETScMatrix constructor 2"); ngs::RegionTimer rt(t);

    ConvertMat();

    /**
       Matrix format is specified.
          AIJ -> SEQAIJ/MPIAIJ
	  BAIJ -> SEQBAIJ/MPIBAIJ
	  IS_AIJ -> local mat to AIJ
	  IS_BAIJ -> local mat to BAIJ
     **/

    PETScMatType pmt;
    MatGetType(petsc_mat, &pmt);
    if (string(pmt) == string(MATIS)) {
      PETScMat loc_mat;
      PETScMatType loc_mt;
      MatISGetLocalMat(petsc_mat, &loc_mat);
      MatGetType(loc_mat, &loc_mt);
      switch(_petsc_mat_type) {
      case AIJ : {
	MatISRestoreLocalMat(petsc_mat, &loc_mat);
	MatConvert(petsc_mat, MATMPIAIJ, MAT_INPLACE_MATRIX, &petsc_mat);
	break;
      }
      case BAIJ : {
	MatISRestoreLocalMat(petsc_mat, &loc_mat);
	MatConvert(petsc_mat, MATMPIBAIJ, MAT_INPLACE_MATRIX, &petsc_mat);
	break;
      }
      case IS_AIJ  : {
	if (loc_mt != string(MATSEQAIJ))
	  { MatConvert(loc_mat, MATSEQAIJ, MAT_INPLACE_MATRIX, &petsc_mat); }
	MatISRestoreLocalMat(petsc_mat, &loc_mat);
	break;
      }
      case IS_BAIJ : {
	if (loc_mt != string(MATSEQBAIJ))
	  { MatConvert(loc_mat, MATSEQBAIJ, MAT_INPLACE_MATRIX, &petsc_mat); }
	MatISRestoreLocalMat(petsc_mat, &loc_mat);
	break;
      }
      default: break;
      }
    }
    else {
      PETScMatType mt = (_petsc_mat_type == AIJ || _petsc_mat_type == IS_AIJ) ? MATSEQAIJ : MATSEQBAIJ;
      if (pmt != mt)
	{ MatConvert(petsc_mat, mt, MAT_INPLACE_MATRIX, &petsc_mat); }
    }
  } // PETScMatrix (..)


  void PETScMatrix :: ConvertMat ()
  {

    static ngs::Timer t("PETScMatrix::ConvertMat"); ngs::RegionTimer rt(t);

    auto parmat = dynamic_pointer_cast<ngs::ParallelMatrix>(ngs_mat);

    bool parallel = parmat != nullptr;

    shared_ptr<ngs::ParallelDofs> row_pardofs = nullptr, col_pardofs = nullptr;
    if (parallel) {
      row_pardofs = parmat->GetRowParallelDofs();
      col_pardofs = parmat->GetColParallelDofs();
    }

    shared_ptr<ngs::BaseSparseMatrix> spmat = dynamic_pointer_cast<ngs::BaseSparseMatrix>( parallel ? parmat->GetMatrix() : ngs_mat);
    if (!spmat) { throw Exception("Can only convert Sparse Matrices to PETSc."); }

    // local PETSc matrix
    PETScMat petsc_mat_loc = CreatePETScMatSeq(spmat, row_subset, col_subset);

    if (parmat && (parmat->GetOpType() == ngs::PARALLEL_OP::C2C))
      { DeleteDuplicateValues(petsc_mat_loc, spmat, row_pardofs, col_pardofs, row_subset, col_subset); }
    
    int bs; MatGetBlockSize(petsc_mat_loc, &bs);

    // Vector conversions
    if (!row_map)
      { row_map = make_shared<NGs2PETScVecMap>(spmat->Width(), bs, row_pardofs, row_subset); }
    if (!col_map)
      { col_map = make_shared<NGs2PETScVecMap>(spmat->Height(), bs, col_pardofs, col_subset); }

    // parallel PETSc matrix
    petsc_mat = parallel ? CreatePETScMatIS (petsc_mat_loc, row_map, col_map) : petsc_mat_loc;

  } // PETScMatrix::ConvertMat()


  void PETScMatrix :: UpdateValues ()
  {
    static ngs::Timer t("PETScMatrix::UpdateValues"); ngs::RegionTimer rt(t);

    // If we have converted the matrix from BAIJ to AIJ, is SetValuesBlocked inefficient??
    // SZ: answer: it shouldn't be inefficient: it expands by blocksize
    // the blocked set of indices and then call MatSetValues (thanks Stefano!)
    auto parmat = dynamic_pointer_cast<ngs::ParallelMatrix>(ngs_mat);
    shared_ptr<BaseMatrix> mat = (parmat == nullptr) ? ngs_mat : parmat->GetMatrix();

    bool update_done = false;
    Iterate<MAX_SYS_DIM>([&](auto n) {
	constexpr int N = 1 + n;
	if constexpr(N==1) {
	    if (auto spmat = dynamic_pointer_cast<ngs::SparseMatrixTM<PETScScalar>>(mat)) {
	      SetPETScMat (petsc_mat, spmat, GetRowMap(), GetColMap());
	      update_done = true;
	    }
	  }
	else {
	  if (auto spmat = dynamic_pointer_cast<ngs::SparseMatrixTM<ngs::Mat<N, N, PETScScalar>>>(mat)) {
	    SetPETScMat (petsc_mat, spmat, GetRowMap(), GetColMap());
	    update_done = true;
	  }
	}
      });
    if (!update_done)
      { throw Exception("Can not update values for this kind of mat!!");}
  } // PETScMatrix :: UpdateValues


  FlatPETScMatrix :: FlatPETScMatrix (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _row_subset,
				      shared_ptr<ngs::BitArray> _col_subset, shared_ptr<NGs2PETScVecMap> _row_map,
				      shared_ptr<NGs2PETScVecMap> _col_map)
    : PETScBaseMatrix(_ngs_mat, _row_subset, _col_subset, _row_map, _col_map)
  {

    static ngs::Timer t("FlatPETScMatrix constructor"); ngs::RegionTimer rt(t);

    // working vectors
    row_hvec = ngs_mat->CreateRowVector();
    col_hvec = ngs_mat->CreateColVector();

    // haha, this is actually the easiest way to find out if a matrix is parallel
    shared_ptr<ngs::ParallelDofs> row_pardofs, col_pardofs;
    if (auto par_row_vec = dynamic_pointer_cast<ngs::ParallelBaseVector>(row_hvec))
      { row_pardofs = par_row_vec->GetParallelDofs(); }
    if (auto par_col_vec = dynamic_pointer_cast<ngs::ParallelBaseVector>(col_hvec))
      { col_pardofs = par_col_vec->GetParallelDofs(); }

    // better take pardofs from created vectors I think (e.g sum-matrix, product-matrix, etc..)
    // if (auto parmat = dynamic_pointer_cast<ngs::ParallelMatrix>(ngs_mat)) {
    //   row_pardofs = parmat->GetRowParallelDofs();
    //   col_pardofs = parmat->GetColParallelDofs();
    // }
    // else if (auto pc = dynamic_pointer_cast<ngs::Preconditioner>(ngs_mat))
    //   { row_pardofs = col_pardofs = pc->GetAMatrix().GetParallelDofs(); }
    // else // can also be a preconditioner, a ProductMatrix, etc..
    //   { row_pardofs = col_pardofs = GetNGsMat()->GetParallelDofs(); }

    bool parallel = row_pardofs != nullptr;

    MPI_Comm comm = (parallel) ? MPI_Comm(row_pardofs->GetCommunicator()) : PETSC_COMM_SELF;

    // Vector conversions
    if (row_map == nullptr) {
	int bs; // this is pretty hacky ... why don't we have this info in NGSolve??
	if (row_map != nullptr)
	  { bs = row_map->GetBS(); }
	else if (parallel)
	  { bs = row_pardofs->GetEntrySize(); }
	else if (ngs_mat->Width())
	  { bs = row_hvec->FVDouble().Size() / ngs_mat->Width(); }
	else // 0 x 0 sequantial matrix ... ffs, just set bs to 1 and hope for the best
	  { bs = 1; }
	row_map = make_shared<NGs2PETScVecMap>(_ngs_mat->Width(), bs, row_pardofs, row_subset);
      }
    if (col_map == nullptr) {
      col_map = ( (row_pardofs == col_pardofs) && (_row_subset == _col_subset) ) ? row_map :
	make_shared<NGs2PETScVecMap>(_ngs_mat->Height(), row_map->GetBS(), col_pardofs, col_subset);
    }

    // Create a Shell matrix, where we have to set function pointers for operations
    // ( the "this" - pointer can be recovered with MatShellGetConext )
    // cout << "MATSHELL " << endl;
    // cout << "nrows " << row_map->GetNRowsLocal() << " " << col_map->GetNRowsLocal() << " " << row_map->GetNRowsGlobal() << " " << col_map->GetNRowsGlobal() << endl;
    // if (row_pardofs != nullptr)
    //   { cout << "pardof- ndofs " << row_pardofs->GetNDofLocal() << " " << row_pardofs->GetNDofGlobal() << " " << col_pardofs->GetNDofLocal() << " " << col_pardofs->GetNDofGlobal() << endl; }
    MatCreateShell (comm, row_map->GetNRowsLocal(), col_map->GetNRowsLocal(), row_map->GetNRowsGlobal(), col_map->GetNRowsGlobal(), (void*) this, &petsc_mat);

    /** Set function pointers **/
    
    // MatMult: y = A * x
    MatShellSetOperation(petsc_mat, MATOP_MULT, (void(*)(void)) this->MatMult);
    
  } // FlatPETScMatrix


  PetscErrorCode FlatPETScMatrix :: MatMult (PETScMat A, PETScVec x, PETScVec y)
  {
    static ngs::Timer t("FlatPETScMatrix MatMult"); ngs::RegionTimer rt(t);

    // y = A * x

    void* ptr; MatShellGetContext(A, &ptr);
    auto& FPM = *( (FlatPETScMatrix*) ptr);

    FPM.GetRowMap()->PETSc2NGs (*FPM.row_hvec, x);

    FPM.ngs_mat->Mult(*FPM.row_hvec, *FPM.col_hvec);

    FPM.GetColMap()->NGs2PETSc(*FPM.col_hvec, y);
    
    return PetscErrorCode(0);
  } // FlatPETScMatrix::MatMult


  MatNullSpace NullSpaceCreate (FlatArray<shared_ptr<ngs::BaseVector>> vecs, shared_ptr<NGs2PETScVecMap> map,
				bool is_orthonormal, bool const_kernel)
  {
    static ngs::Timer t("NullSpaceCreate"); ngs::RegionTimer rt(t);

    Array<PETScVec> petsc_vecs(vecs.Size());
    for (auto k : Range(vecs.Size())) {
      petsc_vecs[k] = map->CreatePETScVector();
      map->NGs2PETSc(*vecs[k], petsc_vecs[k]);
    }
    Array<PETScScalar> dots(vecs.Size());
    if (!is_orthonormal) {
      VecNormalize(petsc_vecs[0], NULL);
      for (int i = 1; i < vecs.Size(); i++) {
	VecMDot(petsc_vecs[i], i, &petsc_vecs[0], &dots[0]);
	for (int j = 0; j < i; j++) dots[j] *= -1.;
	VecMAXPY(petsc_vecs[i], i, &dots[0], &petsc_vecs[0]);
	VecNormalize(petsc_vecs[i], NULL);
      }
    }
    MPI_Comm comm;
    if (auto pds = map->GetParallelDofs())
      { comm = pds->GetCommunicator(); }
    else
      { comm = PETSC_COMM_SELF; }
    MatNullSpace ns; MatNullSpaceCreate(comm, const_kernel ? PETSC_TRUE : PETSC_FALSE, vecs.Size(), &petsc_vecs[0], &ns);
    for (auto v : petsc_vecs) // destroy vecs (reduces reference count by 1)
      { /* VecDestroy(&v); */ }
    return ns;
  } // NullSpaceCreate


  void ExportLinAlg (py::module &m)
  {

    py::class_<PETScBaseMatrix, shared_ptr<PETScBaseMatrix>, ngs::BaseMatrix>
      (m, "PETScBaseMatrix", "Can be used as an NGSolve- or as a PETSc- Matrix")
      .def("SetNullSpace", [](shared_ptr<PETScBaseMatrix> & mat, py::list py_kvecs) {
	  Array<shared_ptr<ngs::BaseVector>> kvecs = makeCArray<shared_ptr<ngs::BaseVector>>(py_kvecs);
	  mat->SetNullSpace(NullSpaceCreate(kvecs, mat->GetRowMap()));
	}, py::arg("kvecs"))
      .def("SetNearNullSpace", [](shared_ptr<PETScBaseMatrix> & mat, py::list py_kvecs) {
	  Array<shared_ptr<ngs::BaseVector>> kvecs = makeCArray<shared_ptr<ngs::BaseVector>>(py_kvecs);
	  mat->SetNearNullSpace(NullSpaceCreate(kvecs, mat->GetRowMap()));
	}, py::arg("kvecs"));
    

    auto pcm = py::class_<PETScMatrix, shared_ptr<PETScMatrix>, PETScBaseMatrix>
      (m, "PETScMatrix", "PETSc matrix, converted from an NGSolve-matrix");

    py::enum_<PETScMatrix::MAT_TYPE>
      (pcm, "MAT_TYPE", docu_string(R"raw_string(
The format to use for the PETSc matrix.
AIJ     .. (parallel) sparse matrix
BAIJ    .. (parallel) sparse block matrix
IS_AIJ  .. sub-assembled diagonal blocks, blocks in sparse matrix format (same as AIJ if not parallel)
IS_BAIJ .. sub-assembled diagonal blocks, blocks in sparse block matrix format (same as BAIJ if not parallel) )raw_string"))
      .value("AIJ"    , PETScMatrix::MAT_TYPE::AIJ)
      .value("BAIJ"   , PETScMatrix::MAT_TYPE::BAIJ)
      .value("IS_AIJ" , PETScMatrix::MAT_TYPE::IS_AIJ)
      .value("IS_BAIJ", PETScMatrix::MAT_TYPE::IS_BAIJ)
      .export_values()
      ;

    pcm.def(py::init<>
	    ([] (shared_ptr<ngs::BaseMatrix> mat, shared_ptr<ngs::BitArray> freedofs,
		 shared_ptr<ngs::BitArray> row_freedofs, shared_ptr<ngs::BitArray> col_freedofs,
		 py::object format)
	     {
	       if (format.is(py::none()))
		 { return make_shared<PETScMatrix> (mat, freedofs ? freedofs : row_freedofs, freedofs ? freedofs : col_freedofs); }
	       else
		 { return make_shared<PETScMatrix> (mat, freedofs ? freedofs : row_freedofs, freedofs ? freedofs : col_freedofs,
						    format.cast<PETScMatrix::MAT_TYPE>()); }
	     }), py::arg("ngs_mat"), py::arg("freedofs") = nullptr, py::arg("row_freedofs") = nullptr, py::arg("col_freedofs") = nullptr,
	    py::arg("format") = py::none());

    pcm.def("dump", [](shared_ptr<PETScMatrix> mat, string fn) {
	cout << "dump mat to >>" << fn << "<<" << endl;
	auto pds = mat->GetRowMap()->GetParallelDofs();
	MPI_Comm comm ( (pds == nullptr) ? PETSC_COMM_SELF : MPI_Comm(pds->GetCommunicator()));
	PetscViewer ov ;
	if (fn.size() != 0)
	  { PetscViewerASCIIOpen(comm,fn.c_str(),&ov); }
	else
	  { ov = PETSC_VIEWER_STDOUT_SELF; }
	MatView(mat->GetPETScMat(), ov);
      }, py::arg("name") = "");

    py::class_<FlatPETScMatrix, shared_ptr<FlatPETScMatrix>, PETScBaseMatrix>
      (m, "FlatPETScMatrix", "A wrapper around an NGSolve-matrix")
      .def(py::init<>
	   ([] (shared_ptr<ngs::BaseMatrix> mat, shared_ptr<ngs::BitArray> freedofs,
		shared_ptr<ngs::BitArray> row_freedofs, shared_ptr<ngs::BitArray> col_freedofs )
	    {
	      return make_shared<FlatPETScMatrix> (mat, freedofs ? freedofs : row_freedofs, freedofs ? freedofs : col_freedofs);
	    }), py::arg("ngs_mat"), py::arg("freedofs") = nullptr, py::arg("row_freedofs") = nullptr, py::arg("col_freedofs") = nullptr);

  }


} // namespace ngs_petsc_interface
