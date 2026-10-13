#!/usr/bin/env python3
"""Capture the CI arm64 benchmark result for a commit into
bench/baselines/arm64.json.

The CI `arm64-test-bench` job publishes the bench JSON as a git note under
refs/notes/bench-arm64 (no auth needed on the read side):

    git fetch origin refs/notes/bench-arm64:refs/notes/bench-arm64

Usage: extract-baseline.py <commit-sha>
Writes the note content verbatim (it is exactly `leanmt-bench --json`
output) to bench/baselines/arm64.json after validating the schema.
See README.md in this directory for the provenance contract.
"""
import json
import subprocess
import sys

NOTES_REF = "bench-arm64"
OUT = "bench/baselines/arm64.json"


def main() -> int:
    sha = sys.argv[1]
    subprocess.run(
        ["git", "fetch", "origin", f"refs/notes/{NOTES_REF}:refs/notes/{NOTES_REF}"],
        check=True,
    )
    note = subprocess.run(
        ["git", "notes", "--ref", NOTES_REF, "show", sha],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(note)  # must be valid JSON
    if data.get("tool") != "leanmt-bench" or "results" not in data:
        print("note is not a leanmt-bench result:", sorted(data), file=sys.stderr)
        return 1
    with open(OUT, "w") as f:
        f.write(note.strip() + "\n")  # verbatim, per the baseline contract
    print(f"wrote {OUT} from refs/notes/{NOTES_REF} @ {sha[:8]} "
          f"({len(data['results'])} length rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
