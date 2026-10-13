#!/usr/bin/env python3
"""Extract the first bench-results.json embedded in a commit comment and
write it to bench/baselines/arm64.json.

Usage: extract-baseline.py <commit-sha>
Reads https://api.github.com/repos/FricoRico/Versta.Leanmt/commits/<sha>/comments
(anonymously; comments are public) and looks for the folded ```json block the
CI `arm64-test-bench` job posts.

Kept for future deliberate baseline refreshes; see README.md in this dir.
"""
import json
import re
import sys
import urllib.request

REPO = "FricoRico/Versta.Leanmt"


def main() -> int:
    sha = sys.argv[1]
    url = f"https://api.github.com/repos/{REPO}/commits/{sha}/comments"
    req = urllib.request.Request(url, headers={"User-Agent": "leanmt-baseline-capture"})
    with urllib.request.urlopen(req) as r:
        comments = json.loads(r.read().decode())
    for c in comments:
        body = c.get("body", "")
        if "leanmt benchmark" not in body:
            continue
        m = re.search(r"```json\n(\{.*?\n\})\n```", body, re.S)
        if not m:
            continue
        data = json.loads(m.group(1))  # validate it is real JSON
        if data.get("tool") != "leanmt-bench" or "results" not in data:
            print("found json block but schema unexpected:", sorted(data), file=sys.stderr)
            return 1
        data.setdefault("provenance", {})
        data["provenance"].update({
            "ci_run_commit": sha,
            "runner": "ubuntu-24.04-arm",
            "captured_by": "bench/baselines/extract-baseline.py",
        })
        out = "bench/baselines/arm64.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"wrote {out} from commit-comment benchmark of {sha[:8]}")
        return 0
    print(f"no benchmark comment found on {sha}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
