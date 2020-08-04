#ifndef FILE_NGSPETSC_TYPEDEFS_HPP
#define FILE_NGSPETSC_TYPEDEFS_HPP

// forward declare PETSc stuff so we can get away without including petsc.h
typedef struct _p_Vec *Vec;
typedef struct _p_Mat *Mat;
typedef struct _p_KSP *KSP;
typedef struct _p_PC *PC;
typedef struct _p_SNES *SNES;
typedef struct _p_IS *IS;
typedef struct _n_PetscOptions *PetscOptions;
typedef struct _p_ISLocalToGlobalMapping* ISLocalToGlobalMapping;
typedef struct _p_MatNullSpace* MatNullSpace;
typedef const char *MatType;
typedef const char *PCType;

namespace ngs_petsc_interface
{

  namespace ngs = ngcomp;
  using ngs::Array, ngs::Range, ngs::FlatArray, ngs::LocalHeap, ngs::BitArray;


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
