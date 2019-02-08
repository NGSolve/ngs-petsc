#include <comp.hpp>
#include <python_ngstd.hpp> 

namespace petsc_if { void NGS_DLL_HEADER ExportPETScInterface(py::module &m); }

PYBIND11_MODULE(ngspetsc, m)
{
  m.attr("__name__") = "NgsPETScInterface";
  petsc_if::ExportPETScInterface(m);
}
