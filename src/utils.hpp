#ifndef FILE_NGSPETSC_UTILS
#define FILE_NGSPETSC_UTILS

namespace ngs_petsc_interface
{

  /** Used to give a name objects which don't have one defined explicitely
      (used to identify options in the petsc options DB) **/
  string GetDefaultId ();

  void SetOptions (FlatArray<string> opts_vals, string prefix = "", PetscOptions opts = NULL);

  void InitializePETSc (FlatArray<string> options);
  
  void FinalizePETSc ();

} // ngs_petsc_interface

#endif
