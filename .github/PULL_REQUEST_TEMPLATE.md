## Summary

<!-- What changes and why. Link the issue. For performance work, note the
expected latency/RSS effect. -->

## Test plan

<!-- Builds/tests run: per-ABI cmake builds, `ctest` with
`-DLEANMT_BUILD_TESTS=ON`, `leanmt-bench` output. Fixture model used. -->

## Checklist

- [ ] Builds warning-clean across the supported NDKs (27.3 / 28.1) — the CI matrix is the gate
- [ ] `ctest` green on host x86_64 (and on arm64 when touching NEON paths)
- [ ] CI benchmark comment reviewed: no unexplained latency/RSS regression at the default config
- [ ] `bench/baselines/` refreshed only as a deliberate, called-out change — never to hide a regression
- [ ] `CHANGELOG.md` updated for behavior, build, or performance changes
- [ ] `third_party/` submodules (ruy, sentencepiece) modified only via an upstream sync, never edited in place
