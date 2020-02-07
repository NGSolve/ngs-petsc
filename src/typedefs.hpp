#ifndef FILE_NGSPETSC_TYPEDEFS_HPP
#define FILE_NGSPETSC_TYPEDEFS_HPP

namespace ngs_petsc_interface
{

#ifndef PETSC_NO_HYPRE
#define PETSC_HAS_HYPRE
#endif

  namespace ngs = ngcomp;
  using ngs::Array;
  using ngs::Range;


  using PETScIS = ::IS;

  using PETScVec = ::Vec;

  using PETScMat = ::Mat;
  using PETScMatType = ::MatType;

  using PETScPC = ::PC;
  using PETScPCType = ::PCType;

  /** Make syntax consistent **/
  using PETScScalar = ::PetscScalar;
  using PETScInt = ::PetscInt;

#ifdef PETSC_INTERFACE_COMPLEX
  static_assert( is_same<PetscScalar, ngs::Complex>::value, "Trying to compile the complex interface with a real PETSc installation, (set -DPETSC_COMPLEX=ON)!");
#else
  static_assert( is_same<PetscScalar, double>::value, "Trying to compile the real interface with a complex PETSc installation, (set -DPETSC_COMPLEX=OFF)!");
#endif

  // static_assert( (is_same<PetscScalar, double>::value || is_same<PetscScalar, ngs::Complex>::value), "Need double or complex PETSc version!");
  // static_assert( (is_same<PetscScalar, double>::value), "Not a double PETSc version!");
  // static_assert( (is_same<PetscScalar, ngs::Complex>::value), "Not a complex PETSc version!");

} // namespace ngs_petsc_interface

#endif
