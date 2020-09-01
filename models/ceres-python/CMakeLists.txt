cmake_minimum_required(VERSION 2.8.12)
project(ceres_python)

#set(PYBIND11_CPP_STANDARD -std=c++14)
#add_subdirectory(pybind11)
find_package(pybind11)
find_package(Ceres)
find_package(Eigen3)
find_package(Sophus)

include_directories(${EIGEN_INCLUDE_DIR}
        ${Sophus_INCLUDE_DIRS}
        )

pybind11_add_module(ceres_python src/direct_tracking.cpp)
#add_executable(ceres_python src/direct_tracking.cpp)
target_link_libraries(ceres_python PRIVATE ceres ${Sophus_LIBS})

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN_INCLUDE_DIR})

# # Build shared library for ceres-solver
#set(BUILD_SHARED_LIBS ON)
#add_subdirectory(lib/external/ceres-solver)
#include_directories(lib/external/ceres-solver/include)
