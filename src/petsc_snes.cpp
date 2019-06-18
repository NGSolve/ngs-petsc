
#include "petsc_interface.hpp"

namespace ngs_petsc_interface
{
  INLINE string name_reason (SNESConvergedReason r) {
    switch(r) {
    case(SNES_CONVERGED_FNORM_ABS     ): return "SNES_CONVERGED_FNORM_ABS";
    case(SNES_CONVERGED_FNORM_RELATIVE): return "SNES_CONVERGED_FNORM_RELATIVE";
    case(SNES_CONVERGED_SNORM_RELATIVE): return "SNES_CONVERGED_SNORM_RELATIVE";
    case(SNES_CONVERGED_ITS           ): return "SNES_CONVERGED_ITS";
    case(SNES_CONVERGED_TR_DELTA      ): return "SNES_CONVERGED_TR_DELTA";
    case(SNES_DIVERGED_FUNCTION_DOMAIN): return "SNES_DIVERGED_FUNCTION_DOMAIN";
    case(SNES_DIVERGED_FUNCTION_COUNT ): return "SNES_DIVERGED_FUNCTION_COUNT";
    case(SNES_DIVERGED_LINEAR_SOLVE   ): return "SNES_DIVERGED_LINEAR_SOLVE";
    case(SNES_DIVERGED_FNORM_NAN      ): return "SNES_DIVERGED_FNORM_NAN";
    case(SNES_DIVERGED_MAX_IT         ): return "SNES_DIVERGED_MAX_IT";
    case(SNES_DIVERGED_LINE_SEARCH    ): return "SNES_DIVERGED_LINE_SEARCH";
    case(SNES_DIVERGED_INNER          ): return "SNES_DIVERGED_INNER";
    case(SNES_DIVERGED_LOCAL_MIN      ): return "SNES_DIVERGED_LOCAL_MIN";
    case(SNES_DIVERGED_DTOL           ): return "SNES_DIVERGED_DTOL";
    case(SNES_DIVERGED_JACOBIAN_DOMAIN): return "SNES_DIVERGED_JACOBIAN_DOMAIN";
    case(SNES_CONVERGED_ITERATING     ): return "SNES_CONVERGED_ITERATING";
    default : return "weird, unknown reason";
    }
  }
  

  PETScSNES :: PETScSNES (shared_ptr<ngs::BilinearForm> _blf, FlatArray<string> _opts, string _name,
			  shared_ptr<ngs::LocalHeap> alh)
    : blf(_blf), use_lh(alh)
  {
    auto pardofs = blf->GetTrialSpace()->GetParallelDofs();
    MPI_Comm comm;
    if (pardofs != nullptr)
      { comm = pardofs->GetCommunicator(); }
    else
      { comm = PETSC_COMM_SELF; }

    auto row_fds = blf->GetTrialSpace()->GetFreeDofs();
    auto col_fds = blf->GetTestSpace()->GetFreeDofs();

    // assemble matrix once so it is allocated
    row_vec = blf->CreateRowVector(); *row_vec = 0;
    blf->AssembleLinearization (*row_vec, *use_lh, false);

    cout << "MAT SIZE: " << blf->GetMatrixPtr()->Height() << " x " << blf->GetMatrixPtr()->Width() << endl;
    
    // Create Matrix for F'(x)
    // jac_mat = make_shared<FlatPETScMatrix> (blf->GetMatrixPtr(), row_fds, col_fds);

    auto trans_mat = make_shared<ngs::Transpose>(blf->GetMatrixPtr());
    trans_mat->SetParallelDofs(blf->GetMatrixPtr()->GetParallelDofs());
    jac_mat = make_shared<FlatPETScMatrix> (trans_mat, row_fds, col_fds);

    // jac_mat = make_shared<FlatPETScMatrix> (blf->GetMatrixPtr(), row_fds, col_fds);

    // jac_mat = make_shared<PETScMatrix> (blf->GetMatrixPtr(), row_fds, col_fds, MATMPIBAIJ);

    // auto width = row_fds->Size();
    // Array<int> perow(width); perow = 1;
    // auto idmat = make_shared<ngs::SparseMatrix<Mat<2>>>(perow);
    // for(auto k : Range(width))
    //   { auto & d = (*idmat)(k,k); d(0,0) = d(1,1) = 1; }
    // jac_mat = make_shared<FlatPETScMatrix> (idmat, row_fds, col_fds);

    // buffer vectors
    col_vec = jac_mat->GetColMap()->CreateNGsVector();
    sol_vec = jac_mat->GetRowMap()->CreatePETScVector();

    cout << "SNES" << endl;
    
    // Create SNES
    SNESCreate(comm, &GetSNES());

    // Get KSP out of SNES and wrap it
    KSP snes_ksp; SNESGetKSP(GetSNES(), &snes_ksp);
    ksp = make_shared<PETScKSP> (jac_mat, snes_ksp);

    // set prefix so we can define unique options for this SNES object
    string name = (_name.size()) ? _name : GetDefaultId(); // name += string("_");
    SNESSetOptionsPrefix(GetSNES(), name.c_str());
    // KSPSetOptionsPrefix(GetKSP()->GetKSP(), name.c_str());

    // Hand given options to global option DB with prefix name
    SetOptions (_opts, name, NULL);

    // Create Vector to hold F(x)
    func_vec = jac_mat->GetRowMap()->CreatePETScVector();

    // Set (non-linear) function evaluation, f = F(x)
    SNESSetFunction(GetSNES(), func_vec, this->EvaluateF, (void*)this);

    // jac_mat2 = make_shared<PETScMatrix> (blf->GetMatrixPtr(), row_fds, col_fds, MATMPIAIJ);
    
    // Set evaluation of the Jacobian
    SNESSetJacobian(GetSNES(), jac_mat->GetPETScMat(), jac_mat->GetPETScMat(), this->EvaluateJac, (void*)this);
  }


  PETScSNES :: ~PETScSNES ()
  {
    SNESDestroy(&GetSNES());
    VecDestroy(&func_vec);
  }


  void PETScSNES :: Finalize ()
  {
    // SNESSetMaxNonlinearStepFailures(GetSNES(), 100);
    
    // Tell the SNES to use options from the global DB (must be called after all other customization)
    SNESSetFromOptions(GetSNES());

    // It is not really necessary to call this
    SNESSetUp(GetSNES());
  }


  void PETScSNES :: Solve (ngs::BaseVector & sol)
  {
    static ngs::Timer tt("PETSc::SNES::Solve - total");
    static ngs::Timer tp("PETSc::SNES::Solve - PETSc");
    RegionTimer rt(tt);

    jac_mat->GetRowMap()->NGs2PETSc(sol, sol_vec);

    // cout << "SNES RHS: " << endl;
    // VecView(sol_vec, PETSC_VIEWER_STDOUT_WORLD);

    {
      RegionTimer rt(tp);
      SNESSolve(GetSNES(), NULL, sol_vec);
    }

    // cout << "SNES SOL: " << sol_vec << endl;
    // VecView(sol_vec, PETSC_VIEWER_STDOUT_WORLD);

    jac_mat->GetRowMap()->PETSc2NGs(sol, sol_vec);

  }


  PetscErrorCode PETScSNES :: EvaluateF (SNES snes, PETScVec x, PETScVec f, void* ctx)
  {
    auto& self = *( (PETScSNES*) ctx);
    HeapReset hr(*self.use_lh);

    // cout << endl << endl << " EVAL F " << endl;
    
    self.jac_mat->GetRowMap()->PETSc2NGs(*self.row_vec, x);

    // self.row_vec->Cumulate(); // OH FOR FUCKS SAKE WHE DO I NEED TO WHY DO I NEED TO DO THIS, IS EVERYTHING BROKEN WITH MPI??
    
    self.blf->ApplyMatrix(*self.row_vec, *self.col_vec, *self.use_lh);

    auto pds = self.jac_mat->GetNGsMat()->GetParallelDofs();
    auto fs = self.jac_mat->GetRowMap()->GetSubSet();
    
    // cout << endl << "-------------" << endl;
    // cout << " ngs - x: " << self.row_vec->GetParallelStatus() << endl;
    // for (auto k : Range(self.row_vec->FVDouble().Size()))
    //   {
    // 	cout << k << ": ";
    // 	if (pds) cout << "(" << pds->IsMasterDof(k/2) << ") ";
    // 	cout << "(" << fs->Test(k/2) << ") ";
    // 	cout << self.row_vec->FVDouble()[k] << endl;
    //   }
    // cout << endl << "-------------" << endl;


    // cout << endl << "-------------" << endl;
    // cout << " I ngs - F(x): " << self.col_vec->GetParallelStatus() << endl;
    // for (auto k : Range(self.col_vec->FVDouble().Size()))
    //   {
    // 	cout << k << ": ";
    // 	if (pds) cout << "(" << pds->IsMasterDof(k/2) << ") ";
    // 	cout << "(" << fs->Test(k/2) << ") ";
    // 	cout << self.col_vec->FVDouble()[k] << endl;
    //   }
    // cout << endl << "-------------" << endl;

    self.jac_mat->GetColMap()->NGs2PETSc(*self.col_vec, f);

    // cout << endl << "-------------" << endl;
    // cout << " II ngs - F(x): " << self.col_vec->GetParallelStatus() << endl;
    // for (auto k : Range(self.col_vec->FVDouble().Size()))
    //   {
    // 	cout << k << ": ";
    // 	if (pds) cout << " (" << pds->IsMasterDof(k/2) << ") ";
    // 	cout << "(" << fs->Test(k/2) << ") ";
    // 	cout << self.col_vec->FVDouble()[k] << endl;
    //   }
    // cout << endl << "-------------" << endl;

    // cout << "x: " << x << endl;
    // VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    // cout << "F(x): " << endl;
    // VecView(f, PETSC_VIEWER_STDOUT_WORLD);

    return PetscErrorCode(0);
  }


  PetscErrorCode PETScSNES :: EvaluateJac (SNES snes, PETScVec x, PETScMat A, PETScMat B, void* ctx)
  {

    auto& self = *( (PETScSNES*) ctx);
    HeapReset hr(*self.use_lh);

    // MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    // MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
    
    // assert(A == self.jac_mat->GetPETScMatrix(), "A != JAC BUFFER, WTF?");
    // if(A != self.jac_mat->GetPETScMat()) { throw Exception("A != JAC BUFFER, WTF?"); }
    // assert(B == self.jac_mat->GetPETScMatrix(), "B != JAC BUFFER, WTF?");
    if(B != self.jac_mat->GetPETScMat()) { throw Exception("B != JAC BUFFER, WTF?"); }

    // KSPSetOperators(self.GetKSP()->GetKSP(), A, B);
    
    auto ksp = self.GetKSP()->GetKSP();
    PETScPC pc; KSPGetPC(ksp, &pc);
    PETScMat amat, pmat; PCGetOperators(pc, &amat, &pmat);
    PetscInt m,n;
    MatGetSize(amat,&m,&n);
    MatGetLocalSize(amat,&m,&n);
    MatGetSize(pmat,&m,&n);
    MatGetLocalSize(pmat,&m,&n);
    
    MatGetSize(self.jac_mat->GetPETScMat(), &m, &n);
    MatGetLocalSize(self.jac_mat->GetPETScMat(), &m, &n);
    
    self.jac_mat->GetRowMap()->PETSc2NGs(*self.row_vec, x);

    // cout << "ASS JAC at " << endl << *self.row_vec << endl;


    // do not re-allocate matrix !
    // cout << "1 JM NGSM: " << self.jac_mat->GetNGsMat() << endl;
    // cout << "1 blf MAT: " << self.blf->GetMatrixPtr() << endl;
    // cout << "mt type " << typeid(*self.blf->GetMatrixPtr()).name() << endl;
    // cout << "MAT BEF: " << endl << *self.blf->GetMatrixPtr() << endl;

    // self.row_vec->Cumulate(); // OH FOR FUCKS SAKE WHE DO I NEED TO WHY DO I NEED TO DO THIS, IS EVERYTHING BROKEN WITH MPI??

    self.blf->AssembleLinearization (*self.row_vec, *self.use_lh, false);
    // cout << "2 JM NGSM: " << self.jac_mat->GetNGsMat() << endl;
    // cout << "2 blf MAT: " << self.blf->GetMatrixPtr() << endl;
    // cout << "mt type " << typeid(*self.blf->GetMatrixPtr()).name() << endl;
    // cout << "MAT AFT: " << endl << *self.blf->GetMatrixPtr() << endl;

    self.jac_mat->UpdateValues();

    return PetscErrorCode(0);
  }


  void ExportSNES (py::module &m)
  {
    extern Array<string> Dict2SA (py::dict & petsc_options);

    py::class_<PETScSNES, shared_ptr<PETScSNES>>
      (m, "SNES", "")
      .def(py::init<>
      	   ([&] (shared_ptr<ngs::BilinearForm> blf, string name, bool finalize, py::dict petsc_options) {
      	     auto opt_array = Dict2SA(petsc_options);
      	     auto snes = make_shared<PETScSNES>(blf, opt_array, name, make_shared<LocalHeap>(10*1024*1024));
      	     if (finalize)
      	       { snes->Finalize(); }
      	     return snes;
      	   }),
      	   py::arg("blf"), py::arg("name") = string(""), py::arg("finalize") = true,
      	   py::arg("petsc_options") = py::dict()
      	   )
      .def("Finalize", [](shared_ptr<PETScSNES> & snes) { snes->Finalize(); })
      .def("Solve", [](shared_ptr<PETScSNES> & snes, shared_ptr<ngs::BaseVector> sol) {
	  snes->Solve(*sol);
	})
      .def("GetKSP", [](shared_ptr<PETScSNES> & snes) -> shared_ptr<PETScKSP> {
	  return snes->GetKSP();
	});
      
  }


} // namespace ngs_petsc_interface
