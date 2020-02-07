project(libpetscinterface)

cmake_minimum_required(VERSION 3.8)

add_ngsolve_python_module(libpetscinterface SHARED
    python_ngspetsc.cpp utils.cpp petsc_linalg.cpp petsc_pc.cpp
    petsc_ksp.cpp petsc_snes.cpp
    slepc_eps.cpp )

include_directories(${PETSC_INCLUDES})

target_link_libraries(libpetscinterface PUBLIC ${PETSC_LIBRARIES})
target_link_libraries(libpetscinterface PUBLIC ${SLEPC_LIBRARIES})
target_include_directories(libpetscinterface PRIVATE ${PETSC_INCLUDES})

install(TARGETS libpetscinterface DESTINATION ${CMAKE_INSTALL_PREFIX}/${MODULE_NAME})
