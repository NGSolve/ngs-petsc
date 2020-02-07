
#include <comp.hpp>
#include "petsc.h"

#ifdef USE_SLEPC
#include "slepc.h"
#endif //  USE_SLEPC

#include <python_ngstd.hpp> 

#include "typedefs.hpp"
#include "utils.hpp"
  
#include "petsc_linalg.hpp"
#include "petsc_pc.hpp"

#include "petsc_ksp.hpp"
#include "petsc_snes.hpp"
