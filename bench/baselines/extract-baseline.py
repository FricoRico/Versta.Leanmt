#!/usr/bin/env python3
"""Capture the CI arm64 benchmark result for a commit into
bench/baselines/.

The CI `arm64-test-bench` job publishes the bench JSON as git notes (no auth
needed on the read side): the canonical word-pool run under
refs/notes/bench-arm64, the numeric-corpus run (digit rows, NEU-30) under
refs/notes/bench-arm64-digits:

    git fetch origin refs/notes/bench-arm64:refs/notes/bench-arm64

Usage: extract-baseline.py <commit-sha> [--digits]
Writes the note content verbatim (it is exactly `leanmt-bench --json`
output) to the matching file under bench/baselines/ after validating the
schema. See README.md in this directory for the provenance contract.
"""
import argparse
import json
import subprocess
import sys

NOTES_REF = "bench-arm64"
OUT = "bench/baselines/arm64.json"
DIGITS_NOTES_REF = "bench-arm64-digits"
DIGITS_OUT = "bench/baselines/arm64-digits.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sha")
    parser.add_argument("--digits", action="store_true",
                        help="extract the numeric-corpus (digit rows) run")
    args = parser.parse_args()
    ref = DIGITS_NOTES_REF if args.digits else NOTES_REF
    out = DIGITS_OUT if args.digits else OUT
    subprocess.run(
        ["git", "fetch", "origin", f"refs/notes/{ref}:refs/notes/{ref}"],
        check=True,
    )
    note = subprocess.run(
        ["git", "notes", "--ref", ref, "show", args.sha],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(note)  # must be valid JSON
    if data.get("tool") != "leanmt-bench" or "results" not in data:
        print("note is not a leanmt-bench result:", sorted(data), file=sys.stderr)
        return 1
    with open(out, "w") as f:
        f.write(note.strip() + "\n")  # verbatim, per the baseline contract
    print(f"wrote {out} from refs/notes/{ref} @ {args.sha[:8]} "
          f"({len(data['results'])} length rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
