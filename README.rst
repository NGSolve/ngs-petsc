
PETSc interface for NGSolve. Work in Progress.
==============================================


Issues when building the library
--------------------

If CMake finds yout PETSc installation, but clatims that it is not working, you can
override this with

```
cmake -DPETSC_EXECUTABLE_RUNS=YES ...
```

If your PETSc installation has beed configured without hypre, you have to tell cmake:


```
cmake -DPETSC_WITH_HYPRE=OFF
```
