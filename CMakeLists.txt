include (ExternalProject)
project(ngspetsc)

cmake_minimum_required(VERSION 3.8)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_modules")

find_package(NGSolve CONFIG REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)

## We cannot set CMAKE_CXX_COMPILER from here, but we can at least throw an error when are using a different
## compiler than the one used to compile NGSolve
get_filename_component(CMCXC ${CMAKE_CXX_COMPILER} ABSOLUTE)
get_filename_component(NGCXC ${NGSOLVE_CXX_COMPILER} ABSOLUTE)
if(NOT ${CMCXC} STREQUAL ${NGCXC})
  message(FATAL_ERROR "Please set CMAKE_CXX_COMPILER to ${NGSOLVE_CXX_COMPILER}, which was used to compile NGSolve.")
endif(NOT ${CMCXC} STREQUAL ${NGCXC})

if(NOT NETGEN_USE_MPI)
  message(FATAL_ERROR "NGSolve version found in ${NGSOLVE_DIR} was not built with MPI support!")
endif(NOT NETGEN_USE_MPI)

find_package(PETSc REQUIRED)

# check if CMAKE_INSTALL_PREFIX is set by user, if not install in NGSolve python dir
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_PYTHON} CACHE PATH "Install dir" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

file(WRITE "dummy.cpp" "int main () { return 0; }")

## figure out if the PETSc library is built with complex or real scalars and 32 or 64 bit indices 
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
try_compile(PETSC_COMPLEX ${CMAKE_BINARY_DIR} "${CMAKE_SOURCE_DIR}/cmake/pcomplex.cpp" CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PETSC_INCLUDES}")
# try_compile(PETSC_LONGIND ${CMAKE_BINARY_DIR} "${CMAKE_SOURCE_DIR}/cmake/plongind.cpp" CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PETSC_INCLUDES}")

message("PETSC_COMPLEX, ${PETSC_COMPLEX}")
# message("PETSC_LONGIND, ${PETSC_LONGIND}")

if (NOT PETSC_COMPLEX)
  # if(NOT PETSC_LONGIND)
    message(STATUS "Compiling the Real PETSc interface!")
    set (MODULE_NAME "ngs_petsc")
  # else(NOT PETSC_LONGIND)
    # message(STATUS "Compiling the Real PETSc interface with 64 bit indices!")
    # set (MODULE_NAME "ngs_petsc_longind")
  # endif(NOT PETSC_LONGIND)
else (NOT PETSC_COMPLEX)
  # if(NOT PETSC_LONGIND)
    message(STATUS "Compiling the Complex PETSc interface!")
    set (MODULE_NAME "ngs_petsc_complex")
  # else(NOT PETSC_LONGIND)
    # message(STATUS "Compiling the Complex PETSc interface with 64 bit indices!")
    # set (MODULE_NAME "ngs_petsc_complex_longind")
  # endif(NOT PETSC_LONGIND)
endif (NOT PETSC_COMPLEX)

add_subdirectory(src)
add_subdirectory(python)
