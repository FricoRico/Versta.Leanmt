# Benchmark harness (latency vs. sentence length, peak RSS) and the batch
# file→file translate CLI (NEU-50, spec `versta.engine-cli/1`) — one binary,
# `leanmt-bench` / `leanmt-bench translate`. Built only when
# LEANMT_BUILD_BENCH=ON (CI sets it; not part of the app build).

add_executable(leanmt-bench bench_translate.cc)
target_link_libraries(leanmt-bench PRIVATE leanmt)
target_compile_options(leanmt-bench PRIVATE -Wall -Wextra)

# End-to-end contract test for the translate mode: line alignment, golden
# output, rerun/batch determinism, CRLF tolerance, exit codes, --version.
# Registered when tests are enabled alongside the bench build; the script
# skips quietly when the pinned fixture model is absent (same policy as the
# unit tests in tests/).
if(LEANMT_BUILD_TESTS)
  if(NOT DEFINED LEANMT_FIXTURES)
    set(LEANMT_FIXTURES
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures/models/ende.student.tiny11")
  endif()
  add_test(
    NAME translate_cli_test
    COMMAND
      ${CMAKE_CURRENT_SOURCE_DIR}/translate_cli_test.sh
      $<TARGET_FILE:leanmt-bench> "${LEANMT_FIXTURES}")
endif()
