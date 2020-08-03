
#include <comp.hpp>
// SZ
// consider removing this include from here
#include "petsc.h"

// SZ
// why you need python stuff in a C++ interface declaration?
#include <python_ngstd.hpp> 

#include "typedefs.hpp"
#include "utils.hpp"
  
#include "petsc_linalg.hpp"
#include "petsc_pc.hpp"

#include "petsc_ksp.hpp"
#include "petsc_snes.hpp"
