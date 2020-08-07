#ifdef PETSc4Py_INTERFACE

#include <pybind11/pybind11.h>
// #include "petsc4py.PETSc_api.h"
#include "petsc4py.h"

template<class PC> class petsc4py_trait {
public:
  // static PC P2C (PyObject* pyob);
  static PyObject* C2P (PC cob);
  // static constexpr const char* tcname = "";
};

#define DECLARE_PB_TYPECASTER(PETSC_CLASS, PYNAME)	\
namespace pybind11 { namespace detail { \
    template<> struct type_caster<PETSC_CLASS> { \
    public: \
    PYBIND11_TYPE_CASTER(PETSC_CLASS, _(PYNAME));   \
    bool load(handle src, bool) { return false; }			\
    static handle cast(PETSC_CLASS pob, return_value_policy, handle) { return petsc4py_trait<PETSC_CLASS>::C2P(pob); } \
    }; \
  }} // namespace pybind11::detail

// template<> petsc4py_trait<ngs_petsc_interface::PETScMat> {
//   PETScMat P2C (PyObject* pyob) { return PyPetscMat_Get(pyob); }
//   PyObject* C2P (PETScMat cob) { return PyPetscMat_New(cob); }
//   string tcname = "PETScMat";
// };
// DECLARE_PB_TYPECASTER(ngs_petsc_interface::PETScMat);

#endif // PETSc4Py_INTERFACE
