# SentencePiece is always built from the bundled submodule.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(SPM_USE_BUILTIN_PROTOBUF ON)
add_subdirectory(sentencepiece EXCLUDE_FROM_ALL)

add_library(spiece_iface INTERFACE)
target_link_libraries(spiece_iface INTERFACE sentencepiece-static)
get_target_property(SentencePiece_INCLUDES sentencepiece-static
                    INCLUDE_DIRECTORIES)
target_include_directories(spiece_iface INTERFACE ${SentencePiece_INCLUDES}
                                                  sentencepiece/src/)

add_library(SentencePiece::SentencePiece ALIAS spiece_iface)

# ruy (the sole int8 GEMM backend)
add_subdirectory(ruy EXCLUDE_FROM_ALL)

# The bundled ruy predates the LLVM change (clang >= 14) that requires
# `-mevex512` alongside the other AVX-512 target flags. Without it the
# x86_64 host build of ruy's AVX-512 dispatch files fails on newer Clang
# (incl. zigcc and Apple Clang). The kernels are runtime-dispatched, so the
# flag only affects compilation of these three translation units.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mevex512" LEANMT_COMPILER_ACCEPTS_MEVEX512)
if(LEANMT_COMPILER_ACCEPTS_MEVEX512)
  foreach(ruy_avx512_target IN ITEMS ruy_kernel_avx512 ruy_pack_avx512
                                     ruy_have_built_path_for_avx512)
    if(TARGET ${ruy_avx512_target})
      target_compile_options(${ruy_avx512_target} PRIVATE -mevex512)
    endif()
  endforeach()
endif()
