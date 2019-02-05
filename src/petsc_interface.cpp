// include order is important here! 
#include <comp.hpp>
// #include "petsc_interface.hpp"
#include "petsc.h"
#include <python_ngstd.hpp> 


namespace petsc_if
{
  using namespace ngcomp;
  
  // void foo() { cout << "hi!!" << endl; }

  INLINE string name_reason (KSPConvergedReason r) {
    switch(r)
      {
      case(KSP_CONVERGED_ITERATING): return "iterating";
      case(KSP_CONVERGED_RTOL_NORMAL): return "rtol_norm";
      case(KSP_CONVERGED_ATOL_NORMAL): return    "atol_norm";
      case(KSP_CONVERGED_RTOL): return "rtol";
      case(KSP_CONVERGED_ATOL): return "atol";
      case(KSP_CONVERGED_ITS): return "its";
      case(KSP_CONVERGED_CG_NEG_CURVE): return "cg_neg";
      case(KSP_CONVERGED_CG_CONSTRAINED): return "cg_constr";
      case(KSP_CONVERGED_STEP_LENGTH): return "step_len";
      case(KSP_CONVERGED_HAPPY_BREAKDOWN): return "happy";
      default: return "not converged";
      }
  }
    
  void STUPID_PETSC_INIT ()
  {
    int argc = 1;
    const char* progname = "whatever..";
    const char* opt = "-info";
    typedef const char * pchar;
    pchar ptrs[2] = { progname, nullptr };
    pchar * pptr = &ptrs[0];
    char** cpptr = (char**)pptr;
    PetscInitialize(&argc, &cpptr, NULL, NULL);
  }


  
  // Maps between ngsolve- and PETSc vector
  class PETSC_VecMap
  {
    shared_ptr<ParallelDofs> pardofs;
    int low, high, loc, glob;
    Array<int> loc_inds, glob_nums;
    Array<double> buf;
  public:
    /**
       Have [low .. high) of PETSc vec. These map to set DOFs of take_dofs.
       _glob .. global numset  (!= pardofs->NGofGlob()!)
    **/
    PETSC_VecMap (int BS, shared_ptr<ParallelDofs> _pardofs,
  		  shared_ptr<BitArray> take_dofs,
  		  int _low, int _high, int _glob)
      : pardofs(_pardofs), low(_low), high(_high), loc(_high-_low), glob(_glob)
    {
      loc_inds.SetSize(loc);
      glob_nums.SetSize(loc);
      buf.SetSize(loc);
      loc = 0;
      for(auto k:Range(pardofs->GetNDofLocal())) {
  	if(!pardofs->IsMasterDof(k)) continue;
  	if(take_dofs && !take_dofs->Test(k)) continue;
  	for(auto kB:Range(BS)) {
  	  loc_inds[loc] = BS*k+kB;
  	  glob_nums[loc] = low+loc;
  	  loc++;
  	}
      }
      if(loc!=(high-low)) {
  	cout << "warning: " << loc << " " << low << " " << high << " " << endl;
  	throw Exception("Inconsistent numbering!");
      }
    }
    PETSC_VecMap (shared_ptr<ParallelDofs> _pardofs,
  		  shared_ptr<BitArray> take_dofs,
  		  int _low, int _high, int _glob)
      : PETSC_VecMap(1, _pardofs, take_dofs, _low, _high, _glob)
    { }
    shared_ptr<ParallelDofs> GetParallelDofs () const { return pardofs; }
    ::Vec CreatePVector()
    {
      ::Vec v;
      VecCreateMPI(pardofs->GetCommunicator(), loc, glob, &v);
      return v;
    };
    // ngsolve -> petsc
    void N2P(shared_ptr<BaseVector> ngs_vec, ::Vec pc_vec)
    {
      ngs_vec->Cumulate();
      VecAssemblyBegin(pc_vec);
      auto fv = ngs_vec->FVDouble();
      for(auto k:Range(loc))
  	buf[k] = fv(loc_inds[k]);
      VecSetValues(pc_vec, loc, &glob_nums[0], &buf[0], INSERT_VALUES);
      VecAssemblyEnd(pc_vec);
    }
    // petsc -> ngsolve
    void P2N(shared_ptr<BaseVector> ngs_vec, ::Vec pc_vec)
    {
      ngs_vec->Distribute();
      VecGetValues(pc_vec, loc, &glob_nums[0], &buf[0]);
      auto fv = ngs_vec->FVDouble();
      fv = 0.0;
      for(auto k:Range(loc))
  	fv(loc_inds[k]) = buf[k];
    }
  };


  template<class TM> INLINE typename mat_traits<TM>::TSCAL* get_ptr(TM & val) { return &val(0,0); }
  template<> INLINE mat_traits<double>::TSCAL* get_ptr<double>(double & val) { return &val; }
  
  template<class TM>
  ::Mat PETSC_SeqBAIJMat (shared_ptr<SparseMatrixTM<TM>> spmat,
  			  shared_ptr<BitArray> take_dofs)
  {
    // TODO: check if TM is square, if not throw exception
    // if(mat_traits<TM>::HEIGHT!=mat_traits<TM>::WIDTH) {
    //   throw Exception("Cannot make non-square PETSC block-matrix!");
    // }
    int BS = mat_traits<TM>::HEIGHT;
    int BSS = BS*BS;
    int NB = spmat->Height();
    int NR = NB * BS;
    Array<int> compress(NB);
    int nb = 0;
    compress = -1;
    for(auto k:Range(NB))
      if(!take_dofs || take_dofs->Test(k)) compress[k] = nb++;
    int nr = nb * BS;
    // cout << "loc BAIJ-Mat, NB " << NB << " " << nb << endl;
    // cout << "BS: " << BS << " " << BSS << endl;
    // cout << "loc BAIJ-Mat, NR " << NR << " " << nr << endl;
    Array<int> nzepr(nb);
    nzepr = 0;
    nb = 0;
    for(auto k:Range(NB)) {
      if(take_dofs && !take_dofs->Test(k)) continue;
      auto & c = nzepr[nb++];
      auto ris = spmat->GetRowIndices(k);
      for(auto j:ris)
  	if(!take_dofs || take_dofs->Test(j)) c++;
    }
    ::Mat petsc_mat;
    MatCreateSeqBAIJ(PETSC_COMM_SELF, BS, nr, nr, -1, &nzepr[0], &petsc_mat); 
    int n_b_entries = 0;
    for(auto k:Range(nb)) n_b_entries += nzepr[k];
    Array<int> cols(n_b_entries);
    n_b_entries = 0;
    for(auto k:Range(NB)) {
      if(take_dofs && !take_dofs->Test(k)) continue;
      auto ris = spmat->GetRowIndices(k);
      for(auto j:ris) if(compress[j]!=-1) cols[n_b_entries++] = compress[j];
    }
    MatSeqBAIJSetColumnIndices(petsc_mat, &cols[0]);
    size_t len_vals = 0;
    for(auto k:Range(NB)) {
      int ck = compress[k];
      if(ck==-1) continue;
      auto ris = spmat->GetRowIndices(k);
      auto rvs = spmat->GetRowValues(k);
      for(auto j:Range(ris.Size())) {
  	int cj = compress[ris[j]];
  	if(cj==-1) continue;
  	double* data = get_ptr(rvs[j]);
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

  
  
  ::Mat PETSC_SEQMat (int BS, shared_ptr<BaseMatrix> mat,
  		      shared_ptr<BitArray> fds)
  {
    ::Mat petsc_mat;
    switch(BS) {
    case(1): {
      auto spm = dynamic_pointer_cast<SparseMatrixTM<double>>(mat);
      // auto cspm = CompressSPM(spm, fds);
      // petsc_mat = PETSC_SeqAIJMat(cspm);
      petsc_mat = PETSC_SeqBAIJMat(spm, fds);
      break;
    }
    case(2): {
      auto spm = dynamic_pointer_cast<SparseMatrixTM<ngbla::Mat<2>>>(mat);
      petsc_mat = PETSC_SeqBAIJMat(spm, fds);
      break;
    }
    case(3): {
      auto spm = dynamic_pointer_cast<SparseMatrixTM<ngbla::Mat<3>>>(mat);
      petsc_mat = PETSC_SeqBAIJMat(spm, fds);
      break;
    }
    case(6): {
      auto spm = dynamic_pointer_cast<SparseMatrixTM<ngbla::Mat<6>>>(mat);
      petsc_mat = PETSC_SeqBAIJMat(spm, fds);
      break;
    }
    default: throw Exception("Cannot make PETSC-mat with that BS!"); break;
    }
    return petsc_mat;
  }

  
  
  ::Mat PETSC_ISMat(int BS, shared_ptr<ParallelDofs> pardofs, 
  		    shared_ptr<BitArray> take_dofs)
  {
    int glob_nd;
    Array<int> globnums;
    pardofs->EnumerateGlobally(take_dofs, globnums, glob_nd);
    int glob_nr = glob_nd * BS;
    MPI_Comm comm = pardofs->GetCommunicator();
    ISLocalToGlobalMapping petsc_map;
    Array<int> compress_globnums(take_dofs ? take_dofs->NumSet() : pardofs->GetNDofLocal());
    // map needs number of local FREE dofs
    int loc_nfd = 0;
    for(auto k:Range(pardofs->GetNDofLocal()))
      if(globnums[k]!=-1)
  	compress_globnums[loc_nfd++] = globnums[k];
    int loc_nfr = loc_nfd * BS;
    ISLocalToGlobalMappingCreate(comm, BS, loc_nfd, &compress_globnums[0], PETSC_COPY_VALUES, &petsc_map);
    // mat needs number for local free AND MASTER dofs for vec ownership
    int loc_nfd_m = 0;
    for(auto k:Range(pardofs->GetNDofLocal()))
      if((!take_dofs || take_dofs->Test(k)) && pardofs->IsMasterDof(k)) loc_nfd_m++;
    int loc_nfr_m = loc_nfd_m * BS;
    ::Mat petsc_mat;
    MatCreateIS(comm, BS, loc_nfr_m, loc_nfr_m, glob_nr, glob_nr, petsc_map, NULL, &petsc_mat);
    return petsc_mat;
  }


  
  ::Mat PETSC_ISMat(shared_ptr<ParallelMatrix> ngs_mat, 
  		    shared_ptr<BitArray> take_dofs)
  { return PETSC_ISMat(1, ngs_mat->GetParallelDofs(), take_dofs); }

  



  // Does not work for compound spaces with different dims (which doesnt work anyways)
  void NGS_KSPSolve (shared_ptr<BilinearForm> blf, shared_ptr<BaseVector> rhs,
  		     shared_ptr<BaseVector> sol, shared_ptr<BitArray> fds,
  		     FlatArray<shared_ptr<BaseVector>> kvecs)
  {
    static Timer t("KSP, total");
    static Timer t_sup("KSP - setup");
    static Timer t_sol("KSP - solve");
    RegionTimer rt(t);
    
    // fds = nullptr;
    auto fes = blf->GetFESpace();
    int BS = fes->GetDimension();
    auto mat = dynamic_pointer_cast<ParallelMatrix>(blf->GetMatrixPtr());
    auto pardofs = mat->GetParallelDofs();

    cout << "KSP for fes, dim = " << BS << endl;
    cout << "ndof " << pardofs->GetNDofLocal() << "  " << pardofs->GetNDofGlobal() << endl;
    cout << "fds: " << fds; if(fds) cout << " set: " << fds->NumSet(); cout << endl;
    
    ::Mat petsc_mat_loc = PETSC_SEQMat(BS, mat->GetMatrix(), fds);
    ::Mat petsc_mat = PETSC_ISMat(BS, pardofs, fds);

    MatISSetLocalMat(petsc_mat, petsc_mat_loc);
    MatAssemblyBegin(petsc_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(petsc_mat, MAT_FINAL_ASSEMBLY);
    
    MatConvert(petsc_mat, MATMPIAIJ, MAT_INPLACE_MATRIX, &petsc_mat);
    if(BS!=1) MatSetBlockSize(petsc_mat, BS);
    // MatConvert(petsc_mat, (BS!=1) ? MATMPIBAIJ : MPIAIJ, MAT_INPLACE_MATRIX, &petsc_mat); //doesnt work with GAMG?

    int row_low, row_high;
    MatGetOwnershipRange(petsc_mat, &row_low, &row_high);
    int glob_nr, glob_nc;
    MatGetSize(petsc_mat, &glob_nr, &glob_nc);
    int col_low = row_low, col_high = row_high;
    
    PETSC_VecMap row_map (BS, mat->GetParallelDofs(), fds, row_low, row_high, glob_nr);
    PETSC_VecMap &col_map(row_map);
    ::Vec petsc_rhs = row_map.CreatePVector(), petsc_sol = col_map.CreatePVector();
    row_map.N2P(rhs, petsc_rhs);

    if(kvecs.Size()) 
      {
  	int dimK = kvecs.Size();
  	Array<::Vec> petsc_kvecs(dimK);
  	for(auto k:Range(dimK)) {
  	  petsc_kvecs[k] = col_map.CreatePVector();
  	  row_map.N2P(kvecs[k], petsc_kvecs[k]);
  	}
  	Array<double> dots(dimK);
  	VecNormalize(petsc_kvecs[0],NULL);
  	for (int i=1; i<dimK; i++) {
  	  /* Orthonormalize vec[i] against vec[0:i-1] */
  	  VecMDot(petsc_kvecs[i],i,&petsc_kvecs[0],&dots[0]);
  	  for (int j=0; j<i; j++) dots[j] *= -1.;
  	  VecMAXPY(petsc_kvecs[i],i,&dots[0],&petsc_kvecs[0]);
  	  VecNormalize(petsc_kvecs[i],NULL);
  	}
  	MyMPI_Barrier(pardofs->GetCommunicator());
  	MatNullSpace petsc_rbm_space;
  	MatNullSpaceCreate(pardofs->GetCommunicator(), PETSC_FALSE, dimK, &petsc_kvecs[0], &petsc_rbm_space);
  	MatSetNearNullSpace(petsc_mat, petsc_rbm_space);
      }    
    
    
    KSP ksp;
    KSPCreate(ngs_comm, &ksp);
    KSPSetOperators(ksp, petsc_mat, petsc_mat); //system-mat, mat for PC
    KSPSetType(ksp, KSPCG);
    PC petsc_prec;
    KSPGetPC(ksp, &petsc_prec);
    // PCSetType(petsc_prec, PCHYPRE);
    PCSetType(petsc_prec, PCGAMG);
    PCGAMGSetType(petsc_prec, PCGAMGAGG);
    {
      RegionTimer rt(t_sup);
      if(MyMPI_GetId(pardofs->GetCommunicator())==0) cout << "KSP setup " << endl;
      KSPSetUp(ksp);
    }
    // rel, abs, div_tol, max_its
    KSPSetTolerances(ksp, 1e-12, 1e-30, PETSC_DEFAULT, 1e3);
    Array<double> resis(1e4);
    resis = 0;
    KSPSetResidualHistory(ksp, &resis[0], 1e4, PETSC_TRUE);
    {
      RegionTimer rt(t_sol);
      if(MyMPI_GetId(pardofs->GetCommunicator())==0) cout << "KSP solve " << endl;
      KSPSolve(ksp, petsc_rhs, petsc_sol);
    }
    int nits;
    KSPGetIterationNumber(ksp, &nits);
    double nres;
    KSPGetResidualNorm(ksp, &nres);
    KSPConvergedReason conv_r;
    KSPGetConvergedReason(ksp, &conv_r);
    
    if(MyMPI_GetId(pardofs->GetCommunicator())==0) {
      cout << "-------" << endl;
      cout << "KSP results: " << endl;
      cout << "KSP errors: " << endl;
      int maxito = min2(nits, 50);
      for(auto k:Range(maxito)) cout << k << " = " << resis[k] << endl; 
      if(maxito<nits) {
  	cout << "...." << endl;
  	cout << nits-1 << " = " << resis[nits-1] << endl; 
      }
      cout << "-------" << endl;
      cout << "KSP converged reason: " << name_reason(conv_r) << endl;
      cout << "KSP needed its: " << nits << endl;
      cout << "KSP err norm: " << nres << endl;
      cout << "KSP err norm (rel): " << nres/resis[0] << endl;
      cout << "-------" << endl;
    }

    col_map.P2N(sol, petsc_sol);

    // cout << "PETSC sol: " << endl;
    // VecView(petsc_sol, PETSC_VIEWER_STDOUT_WORLD);
    // cout << "NGS sol (DISTRIB): " << endl << *sol << endl;; 
    // sol->Cumulate();
    // cout << "NGS sol (CUMUL): " << endl << *sol << endl;; 
  }
  
  void NGS_DLL_HEADER ExportPETScInterface(py::module &m) {
    m.def("KSPSolve",
  	  [](shared_ptr<BilinearForm> blf, shared_ptr<BaseVector> rhs,
  	     shared_ptr<BaseVector> sol, shared_ptr<BitArray> fds,
  	     py::list py_kvecs)
  	  {
  	    Array<shared_ptr<BaseVector>> kvecs = makeCArraySharedPtr<shared_ptr<BaseVector>>(py_kvecs);;
  	    NGS_KSPSolve(blf, rhs, sol, fds, kvecs);
  	  }, py::arg("blf")=nullptr, py::arg("rhs")=nullptr,
  	  py::arg("sol")=nullptr, py::arg("fds")=nullptr,
  	  py::arg("kvecs")=py::list());
    m.def("InitPETSC", [] () { STUPID_PETSC_INIT(); } );
  }

}

