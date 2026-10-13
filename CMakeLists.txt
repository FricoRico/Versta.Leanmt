cmake_minimum_required(VERSION 3.10...3.27)

project(leanmt VERSION 1.0.0 LANGUAGES CXX C)
set(PROJECT_DESCRIPTION "Inference frontend for bergamot models.")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBLITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINE_HIDDEN YES)

option(USE_SSE2 "Use SSE2" OFF)
option(USE_NEON "Use NEON" OFF)

# First-party warning policy. -Werror applies to leanmt's own sources only —
# bundled third_party (ruy, sentencepiece) is compiled with their own (older,
# noisier) flag sets and is explicitly out of our warning budget.
option(LEANMT_STRICT_WARNINGS "Treat leanmt warnings as errors (-Werror)" ON)

set(LEANMT_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wno-unknown-pragmas
    -Wno-missing-field-initializers
    -Wno-unused-function)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND LEANMT_COMPILE_OPTIONS -Wextra-semi)
endif()

if(LEANMT_STRICT_WARNINGS)
  list(APPEND LEANMT_COMPILE_OPTIONS -Werror)
endif()

set(LEANMT_PUBLIC_LIBS "")
set(LEANMT_INTERFACE_LIBS "")
set(LEANMT_PRIVATE_LIBS "")
set(LEANMT_COMPILE_DEFINITIONS "")
set(LEANMT_LINK_OPTIONS "")

add_subdirectory(third_party)

list(APPEND LEANMT_PUBLIC_LIBS SentencePiece::SentencePiece)

list(APPEND LEANMT_COMPILE_DEFINITIONS LEANMT_HAS_RUY)
list(APPEND LEANMT_PRIVATE_LIBS ruy)

if(USE_SSE2)
  list(APPEND LEANMT_COMPILE_DEFINITIONS USE_SSE2)
  list(APPEND LEANMT_COMPILE_OPTIONS -msse2)
endif(USE_SSE2)

if(USE_NEON)
  list(APPEND LEANMT_COMPILE_DEFINITIONS USE_NEON)
endif(USE_NEON)

option(LEANMT_ARM64_TUNE "Enable armv8a dotprod on arm64" OFF)
if(LEANMT_ARM64_TUNE AND ANDROID_ABI STREQUAL "arm64-v8a")
  list(APPEND LEANMT_COMPILE_OPTIONS -march=armv8.2-a+dotprod)
endif()

# cmake-format: off
set(CMAKE_CXX_FLAGS_PROFILE           "${CMAKE_CXX_FLAGS_RELEASE} -pg" CACHE STRING "Flags used by the C++ compiler during profile builds." FORCE)
set(CMAKE_C_FLAGS_PROFILE             "${CMAKE_C_FLAGS_RELEASE} -pg" CACHE STRING "Flags used by the C compiler during profile builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_PROFILE    "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -pg" CACHE STRING "Flags used for linking binaries during profile builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_PROFILE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}-pg" CACHE STRING "Flags used by the shared libraries linker during profile builds." FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_PROFILE
  CMAKE_C_FLAGS_PROFILE
  CMAKE_EXE_LINKER_FLAGS_PROFILE
  CMAKE_SHARED_LINKER_FLAGS_PROFILE
)
# cmake-format: on

message(STATUS "Project name: ${PROJECT_NAME}")
set(PROJECT_VERSION_STRING "${PROJECT_VERSION}")
set(PROJECT_VERSION_STRING_FULL "v${PROJECT_VERSION}")
message(STATUS "Project version: ${PROJECT_VERSION_STRING_FULL}")

add_subdirectory(src)

# Native unit tests (ctest) and the latency/RSS benchmark harness. Both are
# host-only tooling (cross-compiling for Android leaves them off); CI turns
# them on explicitly. Tests download a pinned, checksum-verified FirefoxMT
# tiny model at configure/test time — see tests/fixtures/download-fixtures.sh.
option(LEANMT_BUILD_TESTS "Build the native unit tests (ctest)" OFF)
if(LEANMT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

option(LEANMT_BUILD_BENCH "Build the latency/RSS benchmark harness" OFF)
if(LEANMT_BUILD_BENCH)
  add_subdirectory(bench)
endif()
