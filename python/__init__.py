from . import libpetscinterface

# general utilities
libpetscinterface.__all__ = ["Initialize", "Finalize"]

# linear algebra
libpetscinterface.__all__ += ["PETScBaseMatrix", "PETScMatrix",
                              "FlatPETScMatrix", "VecMap"]

# preconditioners
libpetscinterface.__all__ += ["PETScPrecond", "PETSc2NGsPrecond", "ConvertNGsPrecond", "NGs2PETScPrecond",
                              "HypreAMSPrecond", "FieldSplitPrecond"]

# linear solver
libpetscinterface.__all__ += ["KSP"]

# nmon-linear solver
libpetscinterface.__all__ += ["SNES"]


from .libpetscinterface import *

# this calls Finalize before all PETSc objects can be cleaned up ...
import atexit
atexit.register(libpetscinterface.Finalize)
libpetscinterface.Initialize()
