include (ExternalProject)
project(ngspetsc)

cmake_minimum_required(VERSION 3.8)

option (PETSC_WITH_HYPRE "Does PETSc have hypre enabled?" ON)
option (PETSC_COMPLEX "Are we building the complex version of the interface?" OFF)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_modules")

find_package(NGSolve CONFIG REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)

if(NOT NETGEN_USE_MPI)
  message(FATAL_ERROR "NGSolve version found in ${NGSOLVE_DIR} was not built with MPI support!")
endif(NOT NETGEN_USE_MPI)

find_package(PETSc REQUIRED)

# check if CMAKE_INSTALL_PREFIX is set by user, if not install in NGSolve python dir
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_PYTHON} CACHE PATH "Install dir" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if (NOT PETSC_COMPLEX)
  message(STATUS "Compiling the Real PETSc interface!")
  set (MODULE_NAME "ngs_petsc")
else (NOT PETSC_COMPLEX)
  message(STATUS "Compiling the Complex PETSc interface!")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPETSC_INTERFACE_COMPLEX")
  set (MODULE_NAME "ngs_petsc_complex")
endif (NOT PETSC_COMPLEX)

if(NOT PETSC_WITH_HYPRE)
  message(STATUS "Turning OFF Hypre Auxiliary space Preconditioners.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPETSC_NO_HYPRE")
else(NOT PETSC_WITH_HYPRE)
  message(STATUS "Turning ON Hypre Auxiliary space Preconditioners.")
endif(NOT PETSC_WITH_HYPRE)

add_subdirectory(src)
add_subdirectory(python)
