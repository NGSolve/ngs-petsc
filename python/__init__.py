from . import libpetscinterface

libpetscinterface.__all__ = ['Initialize', 'Finalize', 'PETScBaseMatrix', 'PETScMatrix',
                             'FlatPETScMatrix', 'PETScPrecond', 'NGs2PETScPrecond', 'KSP',
                             'SNES']

from .libpetscinterface import *

# this calls Finalize before all PETSc objects can be cleaned up ...
# import atexit
# atexit.register(libpetscinterface.Finalize)
# libpetscinterface.Initialize()
