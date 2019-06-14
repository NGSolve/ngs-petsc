// include order is important here! 
#include <comp.hpp>
#include "petsc.h"
#include <python_ngstd.hpp> 


namespace ngs_petsc_interface
{
  namespace ngs = ngcomp;
  using ngs::Array;
  using ngs::Range;
  
  INLINE string name_reason (KSPConvergedReason r) {
    switch(r)
      {
      case(KSP_CONVERGED_RTOL_NORMAL    ): return "KSP_CONVERGED_RTOL_NORMAL    ";
      case(KSP_CONVERGED_ATOL_NORMAL    ): return "KSP_CONVERGED_ATOL_NORMAL    ";
      case(KSP_CONVERGED_RTOL           ): return "KSP_CONVERGED_RTOL           ";
      case(KSP_CONVERGED_ATOL           ): return "KSP_CONVERGED_ATOL           ";
      case(KSP_CONVERGED_ITS            ): return "KSP_CONVERGED_ITS            ";
      case(KSP_CONVERGED_CG_NEG_CURVE   ): return "KSP_CONVERGED_CG_NEG_CURVE   ";
      case(KSP_CONVERGED_CG_CONSTRAINED ): return "KSP_CONVERGED_CG_CONSTRAINED ";
      case(KSP_CONVERGED_STEP_LENGTH    ): return "KSP_CONVERGED_STEP_LENGTH    ";
      case(KSP_CONVERGED_HAPPY_BREAKDOWN): return "KSP_CONVERGED_HAPPY_BREAKDOWN";
      case(KSP_DIVERGED_NULL            ): return "KSP_DIVERGED_NULL            ";
      case(KSP_DIVERGED_ITS             ): return "KSP_DIVERGED_ITS             ";
      case(KSP_DIVERGED_DTOL            ): return "KSP_DIVERGED_DTOL            ";
      case(KSP_DIVERGED_BREAKDOWN       ): return "KSP_DIVERGED_BREAKDOWN       ";
      case(KSP_DIVERGED_BREAKDOWN_BICG  ): return "KSP_DIVERGED_BREAKDOWN_BICG  ";
      case(KSP_DIVERGED_NONSYMMETRIC    ): return "KSP_DIVERGED_NONSYMMETRIC    ";
      case(KSP_DIVERGED_INDEFINITE_PC   ): return "KSP_DIVERGED_INDEFINITE_PC   ";
      case(KSP_DIVERGED_NANORINF        ): return "KSP_DIVERGED_NANORINF        ";
      case(KSP_DIVERGED_INDEFINITE_MAT  ): return "KSP_DIVERGED_INDEFINITE_MAT  ";
//       case(KSP_DIVERGED_PC_FAILED       ): return "KSP_DIVERGED_PC_FAILED       "; // NEW
//       case(KSP_DIVERGED_PCSETUP_FAILED  ): return "KSP_DIVERGED_PCSETUP_FAILED  "; // OLD
      case(KSP_CONVERGED_ITERATING      ): return "KSP_CONVERGED_ITERATING      ";
      default:  return "unknown reason??";
      }
  }



  void InitializePETSc (FlatArray<string> options)
  {
    int argc = options.Size()+1;
    const char* progname = "whatever..";
    typedef const char * pchar;
    Array<pchar> ptrs(argc+1);
    ptrs[0] = progname;
    for (auto k : Range(argc-1))
      ptrs[k+1] = options[k].c_str();
    ptrs.Last() = NULL;
    pchar * pptr = &ptrs[0];
    char** cpptr = (char**)pptr;
    PetscInitialize(&argc, &cpptr, NULL, NULL);
  }

  void FinalizePETSc() { PetscFinalize(); }


  
  // Maps between ngsolve- and PETSc vector
  class Ngs2PETScVecMap
  {
    shared_ptr<ngs::ParallelDofs> pardofs;
    size_t low, high, loc, glob;
    Array<PetscInt> loc_inds, glob_nums;
    Array<PetscScalar> buf;
  public:
    /**
       Have [low .. high) of PETSc vec. These map to set DOFs of take_dofs.
       _glob .. global numset  (!= pardofs->NGofGlob()!)
    **/
    Ngs2PETScVecMap (int bs, shared_ptr<ngs::ParallelDofs> _pardofs,
		     shared_ptr<ngs::BitArray> take_dofs,
		     size_t _low, size_t _high, size_t _glob)
      : pardofs(_pardofs), low(_low), high(_high), loc(_high-_low), glob(_glob)
    {
      loc_inds.SetSize(loc);
      glob_nums.SetSize(loc);
      buf.SetSize(loc);
      loc = 0;
      for (auto k : Range(pardofs->GetNDofLocal())) {
  	if(!pardofs->IsMasterDof(k)) continue;
  	if(take_dofs && !take_dofs->Test(k)) continue;
  	for (auto kB : Range(bs)) {
  	  loc_inds[loc] = bs*k+kB;
  	  glob_nums[loc] = low+loc;
  	  loc++;
  	}
      }
      if(loc!=(high-low)) {
  	cout << "warning: " << loc << " " << low << " " << high << " " << endl;
  	throw Exception("Inconsistent numbering!");
      }
    }
    Ngs2PETScVecMap (shared_ptr<ngs::ParallelDofs> _pardofs,
		     shared_ptr<ngs::BitArray> take_dofs,
		     int _low, int _high, int _glob)
      : Ngs2PETScVecMap(1, _pardofs, take_dofs, _low, _high, _glob)
    { }
    shared_ptr<ngs::ParallelDofs> GetParallelDofs () const { return pardofs; }
    ::Vec CreatePETScVec ()
    {
      ::Vec v;
      VecCreateMPI(pardofs->GetCommunicator(), loc, glob, &v);
      return v;
    };
    // ngsolve -> petsc
    void Ngs2PETSc (shared_ptr<ngs::BaseVector> ngs_vec, ::Vec petsc_vec)
    {
      ngs_vec->Cumulate();
      VecAssemblyBegin(petsc_vec);
      auto fv = ngs_vec->FVDouble();
      for (auto k : Range(loc))
  	buf[k] = fv(loc_inds[k]);
      VecSetValues(petsc_vec, loc, &glob_nums[0], &buf[0], INSERT_VALUES);
      VecAssemblyEnd(petsc_vec);
    }
    // petsc -> ngsolve
    void PETSc2Ngs (shared_ptr<ngs::BaseVector> ngs_vec, ::Vec petsc_vec)
    {
      ngs_vec->Distribute();
      VecGetValues(petsc_vec, loc, &glob_nums[0], &buf[0]);
      auto fv = ngs_vec->FVDouble();
      fv = 0.0;
      for (auto k : Range(loc))
  	fv(loc_inds[k]) = buf[k];
    }
  };

  

  template<class TM> INLINE typename ngs::mat_traits<TM>::TSCAL* get_ptr(TM & val) { return &val(0,0); }
  template<> INLINE ngs::mat_traits<double>::TSCAL* get_ptr<double>(double & val) { return &val; }
  
  template<class TM>
  ::Mat CreatePETScMatSeqBAIJ (shared_ptr<ngs::SparseMatrixTM<TM>> spmat,
			       shared_ptr<ngs::BitArray> take_dofs)
  {
    PetscInt bs = ngs::mat_traits<TM>::HEIGHT;
    int bss = bs*bs;
    int nb_tot = spmat->Height();
    int nrows_tot = nb_tot * bs;
    Array<int> compress(nb_tot);
    int nb = 0;
    compress = -1;
    for (auto k : Range(nb_tot))
      if(!take_dofs || take_dofs->Test(k)) compress[k] = nb++;
    PetscInt nrows = nb * bs;
    // cout << "loc BAIJ-Mat, nb_tot " << nb_tot << " " << nb << endl;
    // cout << "bs: " << bs << " " << bsS << endl;
    // cout << "loc BAIJ-Mat, nrows_tot " << nrows_tot << " " << nr << endl;
    Array<PetscInt> nzepr(nb);
    nzepr = 0;
    nb = 0;
    for (auto k : Range(nb_tot)) {
      if(take_dofs && !take_dofs->Test(k)) continue;
      auto & c = nzepr[nb++];
      auto ris = spmat->GetRowIndices(k);
      for (auto j : ris)
  	if(!take_dofs || take_dofs->Test(j)) c++;
    }
    ::Mat petsc_mat;
    MatCreateSeqBAIJ(PETSC_COMM_SELF, bs, nrows, nrows, -1, &nzepr[0], &petsc_mat); 
    int n_b_entries = 0;
    for (auto k : Range(nb)) n_b_entries += nzepr[k];
    Array<PetscInt> cols(n_b_entries);
    n_b_entries = 0;
    for (auto k : Range(nb_tot)) {
      if(take_dofs && !take_dofs->Test(k)) continue;
      auto ris = spmat->GetRowIndices(k);
      for (auto j : ris) if(compress[j]!=-1) cols[n_b_entries++] = compress[j];
    }
    MatSeqBAIJSetColumnIndices(petsc_mat, &cols[0]);
    size_t len_vals = 0;
    for (auto k : Range(nb_tot)) {
      PetscInt ck = compress[k];
      if(ck==-1) continue;
      auto ris = spmat->GetRowIndices(k);
      auto rvs = spmat->GetRowValues(k);
      for (auto j : Range(ris.Size())) {
  	PetscInt cj = compress[ris[j]];
  	if(cj==-1) continue;
  	PetscScalar* data = get_ptr(rvs[j]);
  	MatSetValuesBlocked(petsc_mat, 1, &ck, 1, &cj, data, INSERT_VALUES);
      }
    }
    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);
    // cout << "SPMAT: " << endl << *spmat << endl;
    // cout << "PETSC BLOCK-MAT: " << endl;
    // MatView(petsc_mat, PETSC_VIEWER_STDOUT_SELF);
    return petsc_mat;
  }


  ::Mat CreatePETScMatSeq (shared_ptr<ngs::BaseMatrix> mat,
			   shared_ptr<ngs::BitArray> fds)
  {
    if (auto spm = dynamic_pointer_cast<ngs::SparseMatrixTM<double>>(mat))
      return CreatePETScMatSeqBAIJ(spm, fds);
    if (auto spm = dynamic_pointer_cast<ngs::SparseMatrixTM<ngs::Mat<2>>>(mat))
      return CreatePETScMatSeqBAIJ(spm, fds);
    if (auto spm = dynamic_pointer_cast<ngs::SparseMatrixTM<ngs::Mat<3>>>(mat))
      return CreatePETScMatSeqBAIJ(spm, fds);
    if (auto spm = dynamic_pointer_cast<ngs::SparseMatrixTM<ngs::Mat<4>>>(mat))
      return CreatePETScMatSeqBAIJ(spm, fds);
    if (auto spm = dynamic_pointer_cast<ngs::SparseMatrixTM<ngs::Mat<5>>>(mat))
      return CreatePETScMatSeqBAIJ(spm, fds);
    if (auto spm = dynamic_pointer_cast<ngs::SparseMatrixTM<ngs::Mat<6>>>(mat))
      return CreatePETScMatSeqBAIJ(spm, fds);
    throw Exception("Cannot make PETSc-Mat from this NGSolve-Mat!");
    return ::Mat(NULL);
  }
  
  
  ::Mat CreatePETScMatIS(int abs, shared_ptr<ngs::ParallelDofs> pardofs, 
			 shared_ptr<BitArray> take_dofs)
  {
    PetscInt bs = abs;
    int glob_nd;
    Array<int> globnums;
    pardofs->EnumerateGlobally(take_dofs, globnums, glob_nd);
    PetscInt glob_nrows = glob_nd * bs;
    MPI_Comm comm = pardofs->GetCommunicator();
    Array<PetscInt> compress_globnums(take_dofs ? take_dofs->NumSet() : pardofs->GetNDofLocal());
    // map needs number of local FREE dofs
    PetscInt loc_nfd = 0;
    for (auto k : Range(pardofs->GetNDofLocal()))
      if(globnums[k]!=-1)
  	compress_globnums[loc_nfd++] = globnums[k];
    int loc_nfr = loc_nfd * bs;
    ISLocalToGlobalMapping petsc_map;
    ISLocalToGlobalMappingCreate(comm, bs, loc_nfd, &compress_globnums[0], PETSC_COPY_VALUES, &petsc_map);
    // mat needs number for local free AND MASTER dofs for vec ownership
    int loc_nfd_m = 0;
    for (auto k : Range(pardofs->GetNDofLocal()))
      if((!take_dofs || take_dofs->Test(k)) && pardofs->IsMasterDof(k)) loc_nfd_m++;
    PetscInt loc_nfr_m = loc_nfd_m * bs;
    ::Mat petsc_mat;
    MatCreateIS(comm, bs, loc_nfr_m, loc_nfr_m, glob_nrows, glob_nrows, petsc_map, NULL, &petsc_mat);
    return petsc_mat;
  }



  // Does not work for compound spaces with different dims (which doesnt work anyways)
  py::dict NGS_KSPSolve (shared_ptr<ngs::BaseMatrix> mat, shared_ptr<ngs::BaseVector> rhs,
			 shared_ptr<ngs::BaseVector> sol, shared_ptr<ngs::BitArray> fds,
			 FlatArray<shared_ptr<ngs::BaseVector>> kvecs)
  {
    static ngs::Timer t("KSP, total");
    static ngs::Timer t_sup("KSP - setup");
    static ngs::Timer t_sol("KSP - solve");
    ngs::RegionTimer rt(t);
    
    // fds = nullptr;
    auto parmat = dynamic_pointer_cast<ngs::ParallelMatrix>(mat);
    auto pardofs = mat->GetParallelDofs();
    MPI_Comm comm = pardofs->GetCommunicator();
    
    // cout << "KSP for fes, dim = " << bs << endl;
    // cout << "ndof " << pardofs->GetNDofLocal() << "  " << pardofs->GetNDofGlobal() << endl;
    // cout << "fds: " << fds; if(fds) cout << " set: " << fds->NumSet(); cout << endl;
    
    ::Mat petsc_mat_loc = CreatePETScMatSeq(parmat->GetMatrix(), fds);
    int bs; MatGetBlockSize(petsc_mat_loc, &bs);
    ::Mat petsc_mat = CreatePETScMatIS(bs, pardofs, fds);
    MatISSetLocalMat(petsc_mat, petsc_mat_loc);
    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);
    
    // MatConvert(petsc_mat, MATMPIAIJ, MAT_INPLACE_MATRIX, &petsc_mat);
    // MatConvert(petsc_mat, MATMPIBAIJ, MAT_INPLACE_MATRIX, &petsc_mat);
    MatConvert(petsc_mat, MATMPIAIJ, MAT_INPLACE_MATRIX, &petsc_mat);
    // MatConvert(petsc_mat, (bs!=1) ? MATMPIBAIJ : MPIAIJ, MAT_INPLACE_MATRIX, &petsc_mat); //doesnt work with GAMG?
    if(bs!=1) MatSetBlockSize(petsc_mat, bs);
    
    PetscInt row_low, row_high;
    MatGetOwnershipRange(petsc_mat, &row_low, &row_high);
    PetscInt glob_nr, glob_nc;
    MatGetSize(petsc_mat, &glob_nr, &glob_nc);
    PetscInt col_low = row_low, col_high = row_high;
    
    Ngs2PETScVecMap row_map (bs, mat->GetParallelDofs(), fds, row_low, row_high, glob_nr);
    Ngs2PETScVecMap &col_map(row_map);
    ::Vec petsc_rhs = row_map.CreatePETScVec(), petsc_sol = col_map.CreatePETScVec();
    row_map.Ngs2PETSc(rhs, petsc_rhs);

    MatNullSpace petsc_null_space;
    Array<::Vec> petsc_kvecs;
    if (kvecs.Size()) 
      {
  	int dimK = kvecs.Size();
	petsc_kvecs.SetSize(dimK);
  	for (auto k : Range(dimK)) {
  	  petsc_kvecs[k] = col_map.CreatePETScVec();
  	  row_map.Ngs2PETSc(kvecs[k], petsc_kvecs[k]);
  	}
  	Array<double> dots(dimK);
  	VecNormalize(petsc_kvecs[0],NULL);
  	for (int i = 1; i < dimK; i++) {
  	  /* Orthonormalize vec[i] against vec[0:i-1] */
  	  VecMDot(petsc_kvecs[i],i,&petsc_kvecs[0],&dots[0]);
  	  for (int j = 0; j < i; j++) dots[j] *= -1.;
  	  VecMAXPY(petsc_kvecs[i],i,&dots[0],&petsc_kvecs[0]);
  	  VecNormalize(petsc_kvecs[i],NULL);
  	}
  	MatNullSpaceCreate(comm, PETSC_FALSE, dimK, &petsc_kvecs[0], &petsc_null_space);
  	MatSetNearNullSpace(petsc_mat, petsc_null_space);
  	// MatSetNullSpace(petsc_mat, petsc_null_space);
      }    
    
    
    KSP ksp;
    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, petsc_mat, petsc_mat); //system-mat, mat for PC
    KSPSetFromOptions(ksp);
    // if (kvecs.Size()) 
    //   KSPSetNullSpace(ksp, petsc_null_space);

    {
      ngs::RegionTimer rt(t_sup);
      // if(MyMPI_GetId(comm)==0) cout << "KSP setup " << endl;
      //   KSPSetFromOptions(ksp);
      KSPSetUp(ksp);
    }
    

    PetscScalar rtol, abstol, dtol; PetscInt maxits;
    KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &maxits);
    int nerrs = maxits+1000;
    Array<PetscScalar> errs(nerrs);
    KSPSetResidualHistory(ksp, &errs[0], nerrs, PETSC_TRUE);
    
    {
      ngs::RegionTimer rt(t_sol);
      // if(MyMPI_GetId(comm)==0) cout << "KSP solve " << endl;
      KSPSolve(ksp, petsc_rhs, petsc_sol);
    }

    col_map.PETSc2Ngs(sol, petsc_sol);

    
    auto results = py::dict();
    {
      KSPConvergedReason conv_r; KSPGetConvergedReason(ksp, &conv_r);
      results["conv_r"] = py::str(name_reason(conv_r));
      PetscInt nits; KSPGetIterationNumber(ksp, &nits);
      results["nits"] = py::int_(nits);
      PetscScalar* pr; PetscInt nr; KSPGetResidualHistory(ksp, &pr, &nr); //gets us nr of used res-entries!
      auto py_r_l = py::list(); for (auto k : Range(nr)) py_r_l.append(py::float_(pr[k]));
      results["errs"] = py_r_l;
      PetscScalar res_n; KSPGetResidualNorm(ksp, &res_n);
      results["res_norm"] =  py::float_(res_n);
      PC petsc_prec; KSPGetPC(ksp, &petsc_prec);
      PCType pct; PCGetType(petsc_prec, &pct);
      results["pc_used"] = py::str(string(pct));
    }

    // cout << "PETSC sol: " << endl;
    // VecView(petsc_sol, PETSC_VIEWER_STDOUT_WORLD);
    // cout << "NGS sol (DISTRIB): " << endl << *sol << endl;; 
    // sol->Cumulate();
    // cout << "NGS sol (CUMUL): " << endl << *sol << endl;; 

    return results;
  }

  
  
  void NGS_DLL_HEADER ExportPETScInterface(py::module &m) {
    m.def("KSPSolve",
  	  [](shared_ptr<ngs::BaseMatrix> mat, shared_ptr<ngs::BaseVector> rhs,
  	     shared_ptr<ngs::BaseVector> sol, shared_ptr<BitArray> fds,
  	     py::list py_kvecs, py::kwargs kwargs)
  	  {
	    Array<string> opts;
	    auto ValStr = [&](const auto & Ob) -> string {
	      if (py::isinstance<py::str>(Ob))
		return Ob.template cast<string>();
	      if (py::isinstance<py::bool_>(Ob))
		return Ob.template cast<bool>()==true ? "1" : "0";
	      if (py::isinstance<py::float_>(Ob) ||
		  py::isinstance<py::int_>(Ob))
		return py::str(Ob).cast<string>();
	      return "COULD_NOT_CONVERT";
	    };
	    for (auto item : kwargs) {
	      string name = "-" + item.first.cast<string>();
	      opts.Append(name);
	      // string val = item.second.cast<string>();
	      string val = ValStr(item.second);
	      opts.Append(val);
	    }
	    // cout << "petsc-options: " << endl << opts << endl;
	    // cout << endl << endl;
	    InitializePETSc(opts);
  	    Array<shared_ptr<ngs::BaseVector>> kvecs = makeCArraySharedPtr<shared_ptr<ngs::BaseVector>>(py_kvecs);
	    py::dict results = NGS_KSPSolve(mat, rhs, sol, fds, kvecs);
	    results["petsc_opts"] = py::dict(kwargs);
	    FinalizePETSc();
	    return results;
  	  }, py::arg("mat")=nullptr, py::arg("rhs")=nullptr,
  	  py::arg("sol")=nullptr, py::arg("fds")=nullptr,
  	  py::arg("kvecs")=py::list());
  }

  PYBIND11_MODULE(ngspetsc, m)
  {
    ExportPETScInterface(m);
  }
  
}

