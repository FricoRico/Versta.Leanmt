#!/bin/sh
# End-to-end contract test for `leanmt-bench translate` (NEU-50; versioned
# spec `versta.engine-cli/1`). Exercises the surface the versta.eval harness
# drives as a black box: line-aligned golden output, rerun determinism,
# batch-size invariance, CRLF tolerance, --version, and nonzero exit codes on
# every failure path. Skips quietly when the pinned fixture model is absent
# (same policy as the unit tests).
#
# Usage: translate_cli_test.sh <path-to-leanmt-bench> <fixture-dir>
set -u

BIN="$1"
FIXTURES="$2"

MODEL="$FIXTURES/model.intgemm.alphas.bin"
VOCAB="$FIXTURES/vocab.deen.spm"
SHORTLIST="$FIXTURES/lex.s2t.bin"

if [ ! -f "$MODEL" ]; then
  echo "skip: fixtures not present (run tests/fixtures/download-fixtures.sh)"
  exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

run_translate() {
  # run_translate <input> <output> [extra args...]
  in="$1"; out="$2"; shift 2
  "$BIN" translate --model "$MODEL" --vocab "$VOCAB" --shortlist "$SHORTLIST" \
      --input "$in" --output "$out" "$@" 2>"$WORK/last.stderr"
}

# Goldens pinned by tests/frontend_test.cc against the same fixture model
# (ende.student.tiny11 v2). If a SIMD/dispatch change flips these, it wants a
# human eye — same rule as the unit-test goldens.
cat > "$WORK/input.txt" <<'EOF'
Hello world, this is a small test.
The menu is not available in English.
Where is the central station?
EOF

cat > "$WORK/expected.txt" <<'EOF'
Hallo Welt, das ist ein kleiner Test.
Das Menü ist nicht auf Englisch verfügbar.
Wo ist der Hauptbahnhof?
EOF

# 1. Golden output, line-aligned, exactly one trailing newline at EOF.
run_translate "$WORK/input.txt" "$WORK/out.txt" --threads 2 \
  || fail "translate exited nonzero: $(cat "$WORK/last.stderr")"
cmp -s "$WORK/expected.txt" "$WORK/out.txt" \
  || { diff -u "$WORK/expected.txt" "$WORK/out.txt" >&2; fail "golden output mismatch"; }

# 2. Rerun at a different thread count: byte-identical output (ruy never
#    splits the reduction, so threads must not change bytes).
run_translate "$WORK/input.txt" "$WORK/out_rerun.txt" --threads 1 \
  || fail "rerun exited nonzero"
cmp -s "$WORK/out.txt" "$WORK/out_rerun.txt" \
  || fail "rerun output not byte-identical across thread counts"

# 3. Batch-size invariance: chunking must not change a line's decode.
run_translate "$WORK/input.txt" "$WORK/out_b1.txt" --batch 1 \
  || fail "--batch 1 exited nonzero"
cmp -s "$WORK/out.txt" "$WORK/out_b1.txt" \
  || fail "--batch 1 output differs from default batch"
run_translate "$WORK/input.txt" "$WORK/out_b2.txt" --batch 2 \
  || fail "--batch 2 exited nonzero"
cmp -s "$WORK/out.txt" "$WORK/out_b2.txt" \
  || fail "--batch 2 output differs from default batch"

# 3b. Explicit architecture flags matching the fixture (enc6/dec2, ffn 2,
#     heads 8) keep the golden output byte-identical (NEU-134).
run_translate "$WORK/input.txt" "$WORK/out_arch.txt" \
  --encoder_layers 6 --decoder_layers 2 --ffn_depth 2 --num_heads 8 \
  || fail "explicit matching arch flags exited nonzero"
cmp -s "$WORK/expected.txt" "$WORK/out_arch.txt" \
  || fail "explicit matching arch flags changed the golden output"

# 4. Explicit greedy decode flag is accepted and deterministic.
run_translate "$WORK/input.txt" "$WORK/out_greedy.txt" --beam 1 \
  || fail "--beam 1 exited nonzero"
run_translate "$WORK/input.txt" "$WORK/out_greedy2.txt" --beam 1 \
  || fail "--beam 1 rerun exited nonzero"
cmp -s "$WORK/out_greedy.txt" "$WORK/out_greedy2.txt" \
  || fail "--beam 1 rerun not byte-identical"

# 5. CRLF input tolerated (output identical to LF input).
printf 'Hello world, this is a small test.\r\nWhere is the central station?\r\n' \
  > "$WORK/crlf.txt"
printf 'Hallo Welt, das ist ein kleiner Test.\nWo ist der Hauptbahnhof?\n' \
  > "$WORK/crlf_expected.txt"
run_translate "$WORK/crlf.txt" "$WORK/out_crlf.txt" \
  || fail "CRLF input exited nonzero"
cmp -s "$WORK/crlf_expected.txt" "$WORK/out_crlf.txt" \
  || fail "CRLF input produced different output"

# 6. Failure paths: runtime failures exit 1, usage errors exit 2.
"$BIN" translate --model "$MODEL" --vocab "$VOCAB" --shortlist "$SHORTLIST" \
  --input "$WORK/no-such-file.txt" --output "$WORK/x.txt" 2>/dev/null
code=$?
[ "$code" -eq 1 ] || fail "missing input file: expected exit 1, got $code"

printf 'Hello world, this is a small test.\n\nWhere is the central station?\n' \
  > "$WORK/blank.txt"
"$BIN" translate --model "$MODEL" --vocab "$VOCAB" --shortlist "$SHORTLIST" \
  --input "$WORK/blank.txt" --output "$WORK/x.txt" 2>"$WORK/blank.stderr"
code=$?
[ "$code" -eq 1 ] || fail "blank line: expected exit 1, got $code"
grep -q "blank line 2" "$WORK/blank.stderr" \
  || fail "blank-line error must name the line number: $(cat "$WORK/blank.stderr")"

"$BIN" translate --model "$MODEL" --vocab "$VOCAB" --shortlist "$SHORTLIST" \
  --input "$WORK/input.txt" --output /nonexistent-dir/x.txt 2>/dev/null
code=$?
[ "$code" -eq 1 ] || fail "unwritable output: expected exit 1, got $code"

"$BIN" translate --input "$WORK/input.txt" --output "$WORK/x.txt" 2>/dev/null
code=$?
[ "$code" -eq 2 ] || fail "missing model flags: expected exit 2, got $code"

"$BIN" translate --bogus 2>/dev/null
code=$?
[ "$code" -eq 2 ] || fail "unknown flag: expected exit 2, got $code"

# Architecture mismatch (NEU-134): the fixture is enc6/dec2. Requesting dec4
# must fail fast at model load (exit 1) naming the mismatched field — never
# translate into exit-0 garbage.
"$BIN" translate --model "$MODEL" --vocab "$VOCAB" --shortlist "$SHORTLIST" \
  --input "$WORK/input.txt" --output "$WORK/x.txt" --decoder_layers 4 \
  2>"$WORK/arch.stderr"
code=$?
[ "$code" -eq 1 ] || fail "depth-mismatched arch: expected exit 1, got $code"
grep -q "decoder_layers" "$WORK/arch.stderr" \
  || fail "depth-mismatch error must name the field: $(cat "$WORK/arch.stderr")"

# Zero is a usage error (exit 2), not a valid layer count.
"$BIN" translate --model "$MODEL" --vocab "$VOCAB" --shortlist "$SHORTLIST" \
  --input "$WORK/input.txt" --output "$WORK/x.txt" --decoder_layers 0 \
  2>/dev/null
code=$?
[ "$code" -eq 2 ] || fail "zero architecture value: expected exit 2, got $code"

# 7. --version carries the spec identifier the harness pins against.
"$BIN" --version 2>/dev/null | grep -q "versta.engine-cli/1" \
  || fail "--version output missing spec identifier"

echo "ok - translate CLI contract (golden, determinism, batch invariance, CRLF, exit codes, --version)"
