
PETSc interface for NGSolve. Work in Progress.
==============================================


Issues when building the library
--------------------

If CMake finds yout PETSc installation, but claims that it is not working, you can
override this with

   cmake -DPETSC_EXECUTABLE_RUNS=YES

Specify the installation path using
   -DCMAKE_INSTALL_PREFIX="${NETGENDIR}/lib/python3/dist-packages"
Specify if you want to use  PETSc4PY 
   -DWITH_PETSC4PY=ON
Add PETSc to your LD_LIBRARY_PATH
   export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PETSC_DIR}/${PETSC_ARCH}/lib"

