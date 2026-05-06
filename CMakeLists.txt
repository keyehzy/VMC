cmake_minimum_required(VERSION 3.20)

project(
  vmc
  VERSION 0.1.0
  DESCRIPTION "Variational Monte Carlo experiments"
  LANGUAGES CXX
)

include(FetchContent)

option(VMC_BUILD_TESTS "Build VMC tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR cmake/unused
)
FetchContent_MakeAvailable(eigen)

add_library(vmc_eigen INTERFACE)
add_library(Eigen3::Eigen ALIAS vmc_eigen)
target_include_directories(vmc_eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")

add_library(vmc_core)
target_sources(
  vmc_core
  PRIVATE src/boson_state.cpp
          src/initialization.cpp
          src/lattice.cpp
          src/proposal.cpp
          src/wave_function.cpp
          src/version.cpp
)
target_include_directories(
  vmc_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vmc_core PUBLIC Eigen3::Eigen)

if(MSVC)
  target_compile_options(vmc_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(vmc_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(vmc src/main.cpp)
target_link_libraries(vmc PRIVATE vmc_core)

if(VMC_BUILD_TESTS)
  enable_testing()

  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(googletest)

  add_executable(
    vmc_tests
    tests/boson_state_test.cpp
    tests/initialization_test.cpp
    tests/lattice_test.cpp
    tests/proposal_test.cpp
    tests/wave_function_test.cpp
    tests/version_test.cpp
  )
  target_link_libraries(vmc_tests PRIVATE vmc_core GTest::gtest_main GTest::gmock)

  include(GoogleTest)
  gtest_discover_tests(vmc_tests)
endif()
