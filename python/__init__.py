from . import libpetscinterface

libpetscinterface.__all__ = ['Initialize', 'Finalize', 'PETScBaseMatrix', 'PETScMatrix', 'FlatPETScMatrix', 'PETScPreconditioner', 'NGs2PETSc_PC', 'KSP']

from .libpetscinterface import *

# this calls Finalize before all PETSc objects can be cleaned up ...
# import atexit
# atexit.register(libpetscinterface.Finalize)
# libpetscinterface.Initialize()
