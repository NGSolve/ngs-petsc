project(libpetscinterface)

cmake_minimum_required(VERSION 3.8)

add_ngsolve_python_module(libpetscinterface SHARED
    python_ngspetsc.cpp utils.cpp petsc_linalg.cpp petsc_pc.cpp
    petsc_ksp.cpp petsc_snes.cpp)

include_directories(${PETSC_INCLUDES})

set_target_properties(libpetscinterface PROPERTIES PUBLIC_HEADER "petsc_interface.hpp;petsc_linalg.hpp;petsc_pc.hpp;petsc_ksp.hpp;petsc_snes.hpp;utils.hpp;typedefs.hpp")

target_link_libraries(libpetscinterface PUBLIC ${PETSC_LIBRARIES})
target_include_directories(libpetscinterface PRIVATE ${PETSC_INCLUDES})

install(TARGETS libpetscinterface LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${MODULE_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/${MODULE_NAME}/include )

# install(FILES petsc_interface.hpp petsc_linalg.hpp petsc_pc.hpp petsc_ksp.hpp petsc_snes.hpp utils.hpp typedefs.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/${MODULE_NAME}/include)
