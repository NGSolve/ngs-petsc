#include "petsc_interface.hpp"

namespace ngs_petsc_interface
{

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

  PETScKSP :: PETScKSP (shared_ptr<PETScBaseMatrix> _petsc_mat, FlatArray<string> _opts, string _name)
    : BaseMatrix(_petsc_mat->GetRowMap()->GetParallelDofs()), petsc_mat(_petsc_mat)
  {
    auto comm = _petsc_mat->GetNGsMat()->GetParallelDofs()->GetCommunicator();
    
    // Create KSP
    KSPCreate(comm, &ksp);

    // set prefix so we can define unique options for this KSP object
    string name = (_name.size()) ? _name : GetDefaultId(); name += string("_");
    KSPSetOptionsPrefix(GetKSP(), name.c_str());

    // Hand given options to global option DB with prefix name
    SetOptions (_opts, name, NULL);

    // Set System-Mat, and mat to build the PC from
    KSPSetOperators(GetKSP(), petsc_mat->GetPETScMat(), petsc_mat->GetPETScMat());

    // Tell the KSP to use options from the global DB
    KSPSetFromOptions(GetKSP());

    // Tell PETSc to allocate space to store residual history (per default 1e4) and to reset for each solve
    KSPSetResidualHistory(GetKSP(), NULL, PETSC_DECIDE, PETSC_TRUE);
  }


  PETScKSP :: PETScKSP (shared_ptr<ngs::BaseMatrix> _ngs_mat, shared_ptr<ngs::BitArray> _freedofs, FlatArray<string> _opts, string _name)
    : PETScKSP (make_shared<PETScMatrix> (_ngs_mat, _freedofs, _freedofs, MATMPIAIJ), _opts, _name)
  { ; }

  PETScKSP :: ~PETScKSP ()
    {
      petsc_pc = nullptr;
      petsc_mat = nullptr;
      KSPDestroy(&ksp);
    }


  void PETScKSP :: SetPC (shared_ptr<PETScPreconditioner> apc)
  {
    petsc_pc = apc;
    KSPSetPC(GetKSP(), petsc_pc->GetPETScPC());
  }

  void PETScKSP :: Finalize ()
  {
    static ngs::Timer t("PETSc::KSP::SetUp");
    ngs::RegionTimer rt(t);
    KSPSetUp(GetKSP());

    petsc_rhs = GetMatrix()->GetRowMap()->CreatePETScVector();
    // VecAssemblyBegin(petsc_rhs);
    // VecAssemblyEnd(petsc_rhs);

    petsc_sol = GetMatrix()->GetColMap()->CreatePETScVector();
    // VecAssemblyBegin(petsc_sol);
    // VecAssemblyEnd(petsc_sol);
  }


  void PETScKSP :: Mult (const ngs::BaseVector & x, ngs::BaseVector & y) const
  {
    static ngs::Timer tm("PETSc::KSP::Mult");
    static ngs::Timer ts("PETSc::KSP::Solve");
    ngs::RegionTimer rts(tm);

    GetMatrix()->GetRowMap()->NGs2PETSc(const_cast<ngs::BaseVector&>(x), petsc_rhs);

    {
      ngs::RegionTimer rts(ts);
      KSPSolve(GetKSP(), petsc_rhs, petsc_sol);
    }

    GetMatrix()->GetColMap()->PETSc2NGs(y, petsc_sol);

  }

  
  void ExportKSP (py::module &m)
  {
    auto convert_opts = [](auto & petsc_options) -> Array<string> {
      Array<string> opt_array;
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
      for (auto item : petsc_options)
	{ opt_array.Append(item.first.template cast<string>() + string(" ") + ValStr(item.second)); }
      return opt_array;
    };

    py::class_<PETScKSP, shared_ptr<PETScKSP>, ngs::BaseMatrix>
      (m, "KSP", "")
      .def(py::init<>
	   ([&] (shared_ptr<ngs::BaseMatrix> mat, shared_ptr<ngs::BitArray> freedofs,
		string name, bool finalize, py::dict petsc_options) {
	     auto opt_array = convert_opts(petsc_options);
	     auto ksp = make_shared<PETScKSP>(mat, freedofs, opt_array, name);
	     if (finalize)
	       { ksp->Finalize(); }
	     return ksp;
	   }),
	   py::arg("mat"), py::arg("freedofs") = nullptr,
	   py::arg("name") = string(""), py::arg("finalize") = true,
	   py::arg("petsc_options") = py::dict()
	   )
      .def(py::init<>
	   ([&] (shared_ptr<PETScBaseMatrix> mat, string name, bool finalize, py::dict petsc_options) {
	     auto opt_array = convert_opts(petsc_options);
	     auto ksp = make_shared<PETScKSP>(mat, opt_array, name);
	     if (finalize)
	       { ksp->Finalize(); }
	     return ksp;
	   }),
	   py::arg("mat"), py::arg("name") = string(""), py::arg("finalize") = true,
	   py::arg("petsc_options") = py::dict()
	   )
      .def("GetMatrix", [](shared_ptr<PETScKSP> & ksp) { return ksp->GetMatrix(); } )
      .def("SetPC", [](shared_ptr<PETScKSP> & aksp, shared_ptr<PETScPreconditioner> & apc) {
	  aksp->SetPC(apc);
	})
      .def("Finalize", [](shared_ptr<PETScKSP> & aksp) { aksp->Finalize(); })
      .def_property_readonly("results",
			     [] (PETScKSP & aksp) -> py::dict {
			       KSP ksp = aksp.GetKSP();
			       auto results = py::dict();
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
			       return results;
			     });
  } // ExportKSP

} // namespace ngs_petsc_interface
