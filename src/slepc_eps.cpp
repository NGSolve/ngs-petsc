#ifdef USE_SLEPC

#include "slepc_eps.hpp"

namespace ngs_petsc_interface
{


  INLINE string name_reason (EPSConvergedReason r) {
    switch(r)
      {
      case(EPS_CONVERGED_TOL               ): return "EPS_CONVERGED_TOL";
      case(EPS_CONVERGED_USER              ): return "EPS_CONVERGED_USER";
      case(EPS_DIVERGED_ITS                ): return "EPS_DIVERGED_ITS";
      case(EPS_DIVERGED_BREAKDOWN          ): return "EPS_DIVERGED_BREAKDOWN";
      case(EPS_DIVERGED_SYMMETRY_LOST ): return "EPS_DIVERGED_BREAKDOWN";
      case(EPS_CONVERGED_ITERATING         ): return "EPS_CONVERGED_ITERATING";
      default:  return "unknown reason??";
      }
  }

  SLEPcEPS :: SLEPcEPS (shared_ptr<PETScBaseMatrix> _A, shared_ptr<PETScBaseMatrix> _B, FlatArray<string> _opts, bool _finalize, string _name)
    : A(_A), B(_B)
  {
    auto row_map = A->GetRowMap();
    auto pds = row_map->GetParallelDofs();

    MPI_Comm comm = (pds == nullptr) ? PETSC_COMM_SELF : MPI_Comm(pds->GetCommunicator());

    EPSCreate(comm, &eps);

    SetOptions (_opts, _name, NULL);

    EPSSetOptionsPrefix(eps, _name.c_str());

    EPSSetFromOptions(eps);

    if (B == nullptr)
      { EPSSetOperators(eps, A->GetPETScMat(), PETSC_NULL); }
    else
      { EPSSetOperators(eps, A->GetPETScMat(), B->GetPETScMat()); }

    if (_finalize)
      { Finalize(); }

  } // SLEPcEPS(..)


  SLEPcEPS :: ~SLEPcEPS ()
  {
    // EPSDestroy(eps);
  } // ~SLEPcEPS


  void SLEPcEPS :: Finalize ()
  {
    static ngs::Timer t("SLEPcEPS::Finalize"); RegionTimer rt(t);
    EPSSetUp(eps);
  } // SLEPcEPS::Finalize


  void SLEPcEPS :: Solve ()
  {
    static ngs::Timer t("SLEPcEPS::Solve"); RegionTimer rt(t);
    EPSSolve(eps);
  } // SLEPcEPS::Solve


  EPSConvergedReason SLEPcEPS :: GetConvergedReason ()
  {
    EPSConvergedReason r;
    EPSGetConvergedReason(eps, &r);
    return r;
  } // SLEPcEPS::GetConvergedReason


  bool SLEPcEPS :: IsConverged ()
  {
    return GetConvergedReason() > 0;
  } // SLEPcEPS::IsConverged


  size_t SLEPcEPS :: GetNConvergedEVs ()
  {
    PETScInt nc = -1;
    EPSGetConverged(eps, &nc);
    return nc;
  } // SLEPcEPS::GetNConvergedEVs


  Complex SLEPcEPS :: GetEigenValue (size_t num)
  {
    PETScScalar eigr, eigi;
    EPSGetEigenvalue(eps, num, &eigr, &eigi);
#ifdef PETSC_INTERFACE_COMPLEX
    return eigr;
#else
    return Complex(eigr, eigi);
#endif
  } // SLEPcEPS::GetEigenValue


  Array<Complex> SLEPcEPS :: GetEigenValues ()
  {
    PETScScalar eigr, eigi;
    Array<Complex> out(GetNConvergedEVs());
    for (auto k : Range(out)) {
#ifdef PETSC_INTERFACE_COMPLEX
      EPSGetEigenvalue(eps, k, &out[k], PETSC_NULL);
#else
      EPSGetEigenvalue(eps, k, &eigr, &eigi);
      out[k] = Complex(eigr, eigi);
#endif
    }
    return out;
  } // SLEPcEPS::GetEigenValues
  

  std::tuple<Complex, shared_ptr<ngs::BaseVector>, shared_ptr<ngs::BaseVector>> SLEPcEPS :: GetEigenPair (size_t num)
  {
    auto row_map = A->GetRowMap();
    PETScScalar eigr, eigi;
    PETScVec evr = row_map->CreatePETScVector(), evi = row_map->CreatePETScVector();
    
    std::tuple<Complex, shared_ptr<ngs::BaseVector>, shared_ptr<ngs::BaseVector>> out;

    Complex eval;
    shared_ptr<ngs::BaseVector> nevr = row_map->CreateNGsVector(), nevi = nullptr;

    EPSGetEigenpair(eps, num, &eigr, &eigi, evr, evi); 

#ifdef PETSC_INTERFACE_COMPLEX
    eval = eigr;
#else
    eval = Complex(eigr, eigi);
    nevi = row_map->CreateNGsVector();
    row_map->PETSc2NGs(*nevi, evi);
#endif
    row_map->PETSc2NGs(*nevr, evr);

    VecDestroy(&evr);
    VecDestroy(&evi);

    return make_tuple<Complex, shared_ptr<ngs::BaseVector>, shared_ptr<ngs::BaseVector>>(move(eval), move(nevr), move(nevi));
  } // SLEPcEPS :: GetEigenPair


  Array<std::tuple<Complex, shared_ptr<ngs::BaseVector>, shared_ptr<ngs::BaseVector>>> SLEPcEPS :: GetEigenPairs (size_t num)
  {
    auto row_map = A->GetRowMap();
    PETScScalar eigr, eigi;
    PETScVec evr = row_map->CreatePETScVector(), evi = row_map->CreatePETScVector();

    if (num > GetNConvergedEVs() - 1)
      { cerr << "Asked EPS for too many Eigenpairs!!" << endl; }

    num = min2(GetNConvergedEVs() - 1, num); // must apparently be one less ??

    Array<std::tuple<Complex, shared_ptr<ngs::BaseVector>, shared_ptr<ngs::BaseVector>>> out(num);

    for (auto k : Range(out)) {
      Complex eval;
      shared_ptr<ngs::BaseVector> nevr = row_map->CreateNGsVector(), nevi = nullptr;
      EPSGetEigenpair(eps, num, &eigr, &eigi, evr, evi);
#ifdef PETSC_INTERFACE_COMPLEX
      eval = eigr;
#else
      eval = Complex(eigr, eigi);
      shared_ptr<ngs::BaseVector> nevi = row_map->CreateNGsVector();
      row_map->PETSc2NGs(*nevi, evi);
#endif
      row_map->PETSc2NGs(*nevr, evr);
      out[k] = make_tuple<Complex, shared_ptr<ngs::BaseVector>, shared_ptr<ngs::BaseVector>>(move(eval), move(nevr), move(nevi));
    }

    return out;
  } // SLEPcEPS :: GetEigenPairs

} // namespace ngs_petsc_interface


namespace ngs_petsc_interface
{

  void ExportEPS (py::module & m)
  {
    extern Array<string> Dict2SA (py::dict & petsc_options);

    auto pyeps = py::class_<SLEPcEPS, shared_ptr<SLEPcEPS>> (m, "EPS", "");
    
    pyeps.def(py::init<>
	      ([&] (shared_ptr<PETScBaseMatrix> A, shared_ptr<PETScBaseMatrix> B,
		    string name, bool finalize, py::dict slepc_options) {
		auto opt_array = Dict2SA(slepc_options);
		auto eps = make_shared<SLEPcEPS>(A, B, opt_array, finalize, name);
		return eps;
	      }),
	      py::arg("A"), py::arg("B") = nullptr,
	      py::arg("name") = string(""), py::arg("finalize") = true,
	      py::arg("slepc_options") = py::dict()
	      );
	      
    pyeps.def("Finalize", [&](shared_ptr<SLEPcEPS> & eps) { eps->Finalize(); });

    pyeps.def("Solve", [&](shared_ptr<SLEPcEPS> & eps) { eps->Solve(); });

    pyeps.def_property_readonly("converged", [&](shared_ptr<SLEPcEPS> & eps) -> bool
				{ return eps->IsConverged(); });

    pyeps.def_property_readonly("converged_reason", [&](shared_ptr<SLEPcEPS> & eps) -> string
				{ return name_reason(eps->GetConvergedReason()); });

    pyeps.def_property_readonly("nconv", [&](shared_ptr<SLEPcEPS> & eps) -> size_t
				{ return eps->GetNConvergedEVs(); });

    pyeps.def_property_readonly("evals", [&](shared_ptr<SLEPcEPS> & eps) -> py::list {
				  auto clist = eps->GetEigenValues();
				  auto pylist = MakePyList(clist);
				  return pylist;
				});
    
    pyeps.def("GetEigenPair", [&](shared_ptr<SLEPcEPS> & eps, int num) -> py::tuple {
	auto ctup = eps->GetEigenPair(num);
	py::tuple ptup(3);
	Iterate<3>([&](auto i) { ptup[i.value] = py::cast(get<i.value>(ctup)); });
	return ptup;
      }, py::arg("num") = 0 );

    pyeps.def("GetEigenPairs", [&](shared_ptr<SLEPcEPS> & eps, int num) -> py::list {
	auto ctups = eps->GetEigenPairs(num);
	auto ptups = py::list();
	for (auto k : Range(ctups)) {
	  py::tuple ptup(3);
	  Iterate<3>([&](auto i) { ptup[i.value] = py::cast(get<i.value>(ctups[k])); });
	  ptups.append(ptup);
	}
	return ptups;
      }, py::arg("num") = 0 );

  } // ExportEPS

} // namespace ngs_petsc_interface

#endif // USE_SLEPC
