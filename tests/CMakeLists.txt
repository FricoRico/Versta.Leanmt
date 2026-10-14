# Native unit tests for the leanmt inference core. Tests run on the build
# host (Linux/macOS); validate the same binary paths that ship in the app
# (model mmap load, sentencepiece tokenization, deterministic translate).
#
# Fixtures: tests/fixtures/download-fixtures.sh fetches the pinned
# ende.student.tiny11 model (SHA-256 verified). Model-dependent cases skip
# quietly if fixtures are absent.

if(NOT DEFINED LEANMT_FIXTURES)
  set(LEANMT_FIXTURES
      "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/models/ende.student.tiny11")
endif()

set(LEANMT_TEST_COMPILE_OPTIONS -Wall -Wextra)

function(leanmt_add_test NAME SOURCE)
  add_executable(${NAME} ${SOURCE})
  target_link_libraries(${NAME} PRIVATE leanmt)
  target_compile_options(${NAME} PRIVATE ${LEANMT_TEST_COMPILE_OPTIONS})
  target_compile_definitions(
    ${NAME} PRIVATE LEANMT_FIXTURE_DEFAULT_DIR="${LEANMT_FIXTURES}")
  add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

leanmt_add_test(io_test io_test.cc)
leanmt_add_test(vocabulary_test vocabulary_test.cc)
leanmt_add_test(text_processor_test text_processor_test.cc)
leanmt_add_test(model_test model_test.cc)
leanmt_add_test(frontend_test frontend_test.cc)
leanmt_add_test(shortlist_test shortlist_test.cc)

# Numeric-fidelity gate (NEU-30): a ctest-registered CLI rather than a
# leanmt_test.hh case — it needs its own exit-code contract (failure =
# digit-copy fidelity below the required rate) and report/dump flags for
# golden-diff tables. Default gate: 95% row exact-match on the pinned
# 100-sentence numeric travel corpus.
leanmt_add_test(numeric_fidelity_test numeric_fidelity_test.cc)
target_compile_definitions(
  numeric_fidelity_test
  PRIVATE LEANMT_NUMERIC_DEFAULT_CORPUS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/numeric/travel_ende_100.txt")
