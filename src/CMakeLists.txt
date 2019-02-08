project(ngspetsc)

cmake_minimum_required(VERSION 3.8)

add_ngsolve_python_module(ngspetsc SHARED petsc_interface.cpp)

include_directories(${PETSC_INCLUDES})

target_link_libraries(ngspetsc ${PETSC_LIBRARIES})
target_include_directories(ngspetsc PRIVATE ${PETSC_INCLUDES})

install(TARGETS ngspetsc DESTINATION ${CMAKE_INSTALL_PREFIX})
