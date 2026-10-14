# Branch protection proposal

Applies to `main` (this repo has no `develop`; feature branches merge straight
to `main` via PR).

| Setting | Proposed value | Why |
| --- | --- | --- |
| Require a pull request before merging | on | The maintainer reviews every PR; no direct pushes. |
| Required approvals | 1 | Small team; keep it cheap. |
| Require status checks to pass | on | The CI gate below. |
| Required checks | `ci-gate` | One stable name covering all CI legs (see below). |
| Require branches to be up to date before merging | on | Guarantees the merged result was actually tested by CI. |
| Require conversation resolution | on | Review comments don't get lost. |
| Require signed commits | off | Not worth the setup cost at this stage. |
| Allow force pushes / deletions | off | Protect history. |
| Bypass for admins | off | The gate only works if it also gates its owner. |

Apply via: repo **Settings → Branches → Add rule** for `main`.

Notes for the maintainer (one-time actions):
1. GitHub Actions is enabled for the repo (default for public repos).
2. Add `ci-gate` as the *required* check only after its first workflow run
   has completed on `main`, otherwise GitHub cannot find the name in the
   picker.
3. Require only `ci-gate`, not the individual legs. Required checks are
   matched by exact job name, and the build legs embed matrix values
   (`android arm64-v8a / NDK 27.3.13750724`, …), so any future ABI/NDK matrix
   edit would silently rename them and un-gate `main`. `ci-gate` aggregates:
   - `android-build` — warning-gated compile matrix: 4 ABIs × 2 NDKs
   - `host-test` — ctest suite on host x86_64 (SSE2 path)
   - `arm64-test-bench` — ctest + latency/RSS benchmark on a native arm64
     runner (NEON path); results published as a PR/commit comment
4. No secrets are needed: builds and tests are self-contained and publish
   results as comments and git notes.
