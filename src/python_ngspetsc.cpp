
#include "petsc_interface.hpp"

namespace ngs_petsc_interface
{

  extern void ExportUtils (py::module &m);
  extern void ExportLinAlg (py::module &m);
  extern void ExportKSP (py::module &m);
  extern void ExportPC (py::module & m);
  
  PYBIND11_MODULE(libpetscinterface, m)
  {
    ExportUtils(m);
    ExportLinAlg(m);
    ExportPC(m);
    ExportKSP(m);
  }
  
} // namespace ngs_petsc_interface
