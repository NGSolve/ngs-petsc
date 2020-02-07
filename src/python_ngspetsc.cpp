
#include "petsc_interface.hpp"

namespace ngs_petsc_interface
{

  Array<string> Dict2SA (py::dict & petsc_options) {
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

  
  extern void ExportUtils (py::module &m);
  extern void ExportLinAlg (py::module &m);
  extern void ExportPC (py::module & m);
  extern void ExportKSP (py::module &m);
  extern void ExportSNES (py::module &m);

#ifdef USE_SLEPC
  extern void ExportEPS (py::module & m);
#endif
  
  PYBIND11_MODULE(libpetscinterface, m)
  {
    static size_t ngp_hs = 1024*1024*10;
    static shared_ptr<LocalHeap> ngp_lh = make_shared<LocalHeap>(ngp_hs, "NGs-PETSc Interface lh", true);

    m.def("SetHeapSize",
	  [](size_t ahs) {
	    if (ahs > ngp_hs) {
	      ngp_hs = ahs;
	      ngp_lh = make_shared<LocalHeap>(ngp_hs, "NGs-PETSc Interface lh", true);
	    }
	  }, py::arg("size"));

    ExportUtils(m);
    ExportLinAlg(m);
    ExportPC(m);
    ExportKSP(m);
    ExportSNES(m);

#ifdef USE_SLEPC
    ExportEPS (m);
#endif
  }
  
} // namespace ngs_petsc_interface
