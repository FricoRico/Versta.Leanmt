#!/usr/bin/env bash
# Downloads and verifies the tiny FirefoxMT (bergamot) model used by leanmt's
# native tests and benchmarks. The artifact is pinned by SHA-256 so CI results
# are comparable across runs and machines.
#
# Usage:
#   tests/fixtures/download-fixtures.sh [dest-dir]
# Default dest-dir is <this script's dir>/models/ende.student.tiny11.
set -euo pipefail

NAME="ende.student.tiny11"
URL="https://data.statmt.org/bergamot/models/deen/ende.student.tiny11.v2.93821e13b3c511b5.tar.gz"
SHA256="93821e13b3c511b5390f9fb79f476f738d66e3656889bf076e906897e614fed2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-${LEANMT_FIXTURES:-${SCRIPT_DIR}/models/${NAME}}}"
TARBALL="${DEST}/../../${NAME}.tar.gz"

if [[ -f "${DEST}/model.intgemm.alphas.bin" && -f "${DEST}/vocab.deen.spm" && -f "${DEST}/lex.s2t.bin" && -f "${DEST}/.checked" ]]; then
  echo "fixtures present at ${DEST}, skipping download"
  exit 0
fi

mkdir -p "${DEST}"
echo "downloading ${URL}"
curl -fL --retry 3 --retry-delay 2 -o "${TARBALL}" "${URL}"
echo "${SHA256}  ${TARBALL}" | sha256sum -c -

tar -xzf "${TARBALL}" -C "$(dirname "${DEST}")" \
  "${NAME}/model.intgemm.alphas.bin" \
  "${NAME}/vocab.deen.spm" \
  "${NAME}/lex.s2t.bin"

rm -f "${TARBALL}"
touch "${DEST}/.checked"
echo "fixtures ready at ${DEST}"
