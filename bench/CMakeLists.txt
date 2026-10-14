# Benchmark harness: latency vs. sentence length and peak RSS.
# Built only when LEANMT_BUILD_BENCH=ON (CI sets it; not part of the app build).

add_executable(leanmt-bench bench_translate.cc)
target_link_libraries(leanmt-bench PRIVATE leanmt)
target_compile_options(leanmt-bench PRIVATE -Wall -Wextra)
