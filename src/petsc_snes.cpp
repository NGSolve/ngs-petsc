
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
			  shared_ptr<ngs::LocalHeap> _lh, JACOBI_MAT_MODE _jac_mode)
    : blf(_blf), use_lh(_lh), mode(_jac_mode)
  {

    if (use_lh == nullptr)
      { use_lh = make_shared<ngs::LocalHeap>(10*1024*1024); }

    auto pardofs = blf->GetTrialSpace()->GetParallelDofs();
    MPI_Comm comm;
    if (pardofs != nullptr)
      { comm = pardofs->GetCommunicator(); }
    else
      { comm = PETSC_COMM_SELF; }

    auto row_fds = blf->GetTrialSpace()->GetFreeDofs();
    auto col_fds = blf->GetTestSpace()->GetFreeDofs();

    // linearization vector
    lin_vec = blf->CreateRowVector();

    // Create Matrix for F'(x)
    // jac_mat = make_shared<FlatPETScMatrix> (blf->GetMatrixPtr(), row_fds, col_fds);

    // auto trans_mat = make_shared<ngs::Transpose>(blf->GetMatrixPtr());
    // trans_mat->SetParallelDofs(blf->GetMatrixPtr()->GetParallelDofs());
    // jac_mat = make_shared<FlatPETScMatrix> (trans_mat, row_fds, col_fds);

    // Vector maps
    auto trs = blf->GetTrialSpace();
    auto row_map = make_shared<NGs2PETScVecMap>(trs->GetNDof(), trs->GetDimension(), trs->GetParallelDofs(), trs->GetFreeDofs());
    auto tss = blf->GetTrialSpace();
    auto col_map = make_shared<NGs2PETScVecMap>(tss->GetNDof(), tss->GetDimension(), tss->GetParallelDofs(), tss->GetFreeDofs());

    if (mode == APPLY) {
      auto lap = make_shared<ngs::LinearizedBilinearFormApplication>(blf, lin_vec.get(), *use_lh);
      jac_mat = make_shared<FlatPETScMatrix> (lap, nullptr, nullptr, row_map, col_map);
    }
    else {
      // assemble matrix once so it is allocated
      *lin_vec = 0;
      blf->AssembleLinearization (*lin_vec, *use_lh, false);
      if (mode == FLAT)
	{ jac_mat = make_shared<FlatPETScMatrix> (blf->GetMatrixPtr(), row_fds, col_fds, row_map, col_map); }
      else if (mode == CONVERT) // IS makes UpdateValues easier
	{ jac_mat = make_shared<PETScMatrix> (blf->GetMatrixPtr(), row_fds, col_fds, MATMPIBAIJ, row_map, col_map); }
    }

    // buffer vectors
    row_vec = jac_mat->GetRowMap()->CreateNGsVector();
    col_vec = jac_mat->GetColMap()->CreateNGsVector();
    sol_vec = jac_mat->GetRowMap()->CreatePETScVector();

    // Create SNES
    SNESCreate(comm, &GetSNES());

    // Get KSP out of SNES and wrap it
    KSP snes_ksp; SNESGetKSP(GetSNES(), &snes_ksp);
    ksp = make_shared<PETScKSP> (jac_mat, snes_ksp);

    // set prefix so we can define unique options for this SNES object
    string name = (_name.size()) ? _name : GetDefaultId();
    SNESSetOptionsPrefix(GetSNES(), name.c_str());
    // KSPSetOptionsPrefix(GetKSP()->GetKSP(), name.c_str());

    // Hand given options to global option DB with prefix name
    SetOptions (_opts, name, NULL);

    // Create Vector to hold F(x)
    func_vec = jac_mat->GetRowMap()->CreatePETScVector();

    // Set (non-linear) function evaluation, f = F(x)
    SNESSetFunction(GetSNES(), func_vec, this->EvaluateF, (void*)this);

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

    self.jac_mat->GetRowMap()->PETSc2NGs(*self.row_vec, x);

    self.blf->ApplyMatrix(*self.row_vec, *self.col_vec, *self.use_lh);

    self.jac_mat->GetColMap()->NGs2PETSc(*self.col_vec, f);

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
    
    // actually, this happens if snes_mf or snes_mf_operator are set
    // if(A != self.jac_mat->GetPETScMat()) { throw Exception("A != JAC BUFFER, WTF?"); }
    if(B != self.jac_mat->GetPETScMat()) { throw Exception("B != JAC BUFFER, WTF?"); }
    
    self.jac_mat->GetRowMap()->PETSc2NGs(*self.lin_vec, x);

    // do not re-allocate matrix !
    if (self.mode != APPLY)
      { self.blf->AssembleLinearization (*self.lin_vec, *self.use_lh, false); }

    cout << "UPDATE VALUES!" << endl;
    self.jac_mat->UpdateValues();
    cout << "UPDATE VALUES DONE!" << endl;

    return PetscErrorCode(0);
  }


  void ExportSNES (py::module &m)
  {
    extern Array<string> Dict2SA (py::dict & petsc_options);

    auto snes = py::class_<PETScSNES, shared_ptr<PETScSNES>>
      (m, "SNES", "");

    py::enum_<PETScSNES::JACOBI_MAT_MODE>
      (snes, "JACOBI_MAT_MODE", docu_string(R"raw_string(
How the jacobi matrix should be implemented:
APPLY   ... Do not assemble jacobi mat (use ApplyLinearization)
FLAT    ... Assemble Jacobi matrix, but only wrap it to PETSc
CONVERT ... Assemble Jacobi matrix, and convert it to a PETSc matrix)raw_string"))
      .value("APPLY"  , PETScSNES::JACOBI_MAT_MODE::APPLY)
      .value("FLAT"   , PETScSNES::JACOBI_MAT_MODE::FLAT)
      .value("CONVERT", PETScSNES::JACOBI_MAT_MODE::CONVERT)
      .export_values()
      ;

    snes.def(py::init<>
	     ([&] (shared_ptr<ngs::BilinearForm> blf, string name, bool finalize,
		   PETScSNES::JACOBI_MAT_MODE mode, py::dict petsc_options) {
	       auto opt_array = Dict2SA(petsc_options);
	       auto snes = make_shared<PETScSNES>(blf, opt_array, name, make_shared<LocalHeap>(10*1024*1024), mode);
	       if (finalize)
		 { snes->Finalize(); }
	       return snes;
	     }),
	     py::arg("blf"), py::arg("name") = string(""), py::arg("finalize") = true,
	     py::arg("mode") = PETScSNES::JACOBI_MAT_MODE::FLAT, py::arg("petsc_options") = py::dict()
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
