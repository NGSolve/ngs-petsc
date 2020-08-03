
PETSc interface for NGSolve. Work in Progress.
==============================================


Issues when building the library
--------------------

If CMake finds yout PETSc installation, but claims that it is not working, you can
override this with

   cmake -DPETSC_EXECUTABLE_RUNS=YES

