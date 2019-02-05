project(ngspetsc)

cmake_minimum_required(VERSION 3.5)

add_library(ngspetsc SHARED petsc_interface.cpp )

include_directories(${PETSC_INCLUDES})

target_include_directories(ngspetsc PRIVATE ${PETSC_INCLUDES})
target_link_libraries(ngspetsc PUBLIC ngcomp ngstd ngla)

install(TARGETS ngspetsc DESTINATION ${CMAKE_INSTALL_PREFIX})
