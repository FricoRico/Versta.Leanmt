# Pack acceptance probe: loads one bundle direction end-to-end and translates
# a probe corpus under app-parity decode settings (NEU-60). Built only when
# LEANMT_BUILD_PROBE=ON (the Versta.Models pack-acceptance CI sets it; not part
# of the app build).

add_executable(leanmt-probe probe_pack.cc)
target_link_libraries(leanmt-probe PRIVATE leanmt)
target_compile_options(leanmt-probe PRIVATE -Wall -Wextra)
