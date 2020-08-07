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

template<class C> struct pbholder
{
  pbholder () = default;
  pbholder(C _value) : value(_value) { ; }
  operator pbholder () { return value; }
  C value;
};

// #define DECLARE_PB_TYPECASTER(PETSC_CLASS, PYNAME)	\
// namespace pybind11 { namespace detail { \
//     template<> struct type_caster<PETSC_CLASS> { \
//     public: \
//     PYBIND11_TYPE_CASTER(PETSC_CLASS, _(PYNAME));   \
//     bool load(handle src, bool) { return false; }			\
//     static handle cast(PETSC_CLASS pob, return_value_policy, handle) { return petsc4py_trait<PETSC_CLASS>::C2P(pob); } \
//     }; \
//   }} // namespace pybind11::detail

// template<> petsc4py_trait<ngs_petsc_interface::PETScMat> {
//   PETScMat P2C (PyObject* pyob) { return PyPetscMat_Get(pyob); }
//   PyObject* C2P (PETScMat cob) { return PyPetscMat_New(cob); }
//   string tcname = "PETScMat";
// };
// DECLARE_PB_TYPECASTER(ngs_petsc_interface::PETScMat);

#define DECLARE_PB_TYPECASTER(CCLASS, PYCLASS, C2P, P2C, PYNAME)	\
  template class pbholder<CCLASS>;					\
  namespace pybind11 { namespace detail {				\
      template<> struct type_caster<pbholder<CCLASS>> {			\
      public:								\
      PYBIND11_TYPE_CASTER(pbholder<CCLASS>, _(PYNAME));		\
      bool load(handle src, bool) {					\
        PyObject *py_src = src.ptr();					\
        if (PyObject_TypeCheck(py_src, &PYCLASS)) {			\
          value.value = P2C(py_src);					\
        } else {							\
          return false;							\
        }								\
        return !PyErr_Occurred();					\
      }									\
      static handle cast(pbholder<CCLASS> pob, return_value_policy, handle) { return C2P(pob.value); } \
      };								\
    }} // namespace pybind11::detail

#define DECLARE_PB_CONVERTIBLE(CCLASS) py::implicitly_convertible<pbholder<CCLASS>, CCLASS>();
  


#endif // PETSc4Py_INTERFACE
