#!/usr/bin/env bash
# Downloads and verifies the pinned FirefoxMT (bergamot) models used by
# leanmt's native tests and benchmarks. Artifacts are pinned by SHA-256 so
# CI results are comparable across runs and machines.
#
# Fixtures:
#
#   ende.student.tiny11  — shared-vocab reference model (single vocab.deen.spm,
#                          tied Wemb). statsmt bergamot mirror, v2 tarball.
#   enja.base.v2.3       — two-vocabulary model (separate srcvocab/trgvocab,
#                          encoder_Wemb/decoder_Wemb tensors): English→Japanese
#                          as shipped by Firefox Translations remote-settings
#                          v2.3. Exercises the decoder_Wemb_intgemm8 alias and
#                          the decoder_Wemb_QuantMultA alpha path that a
#                          single-vocab fixture never touches.
#
# Usage:
#   tests/fixtures/download-fixtures.sh [ende-dir] [enja-dir]
# Defaults are <this script's dir>/models/{ende.student.tiny11,enja.base.v2.3}.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

ENDE_NAME="ende.student.tiny11"
ENDE_URL="https://data.statmt.org/bergamot/models/deen/ende.student.tiny11.v2.93821e13b3c511b5.tar.gz"
ENDE_SHA256="93821e13b3c511b5390f9fb79f476f738d66e3656889bf076e906897e614fed2"

ENDE_DIR="${1:-${LEANMT_FIXTURES:-${MODELS_DIR}/${ENDE_NAME}}}"
ENJA_DIR="${2:-${LEANMT_FIXTURES_2V:-${MODELS_DIR}/enja.base.v2.3}}"

# English→Japanese v2.3, served from Mozilla's remote-settings attachment CDN
# (the same artifacts Firefox desktop downloads in-app). Hashes are the
# remote-settings attachment sha256 for each file.
ENJA_MODEL_URL="https://firefox-settings-attachments.cdn.mozilla.net/main-workspace/translations-models/4e19636e-4644-4072-8a05-9ef695d4a3ae.bin"
ENJA_MODEL_SHA256="59ae659f9bb63e4f81f474fe3c03d3f4499434b5f9e779fab7c12a45f31fd562"
ENJA_SRCVOCAB_URL="https://firefox-settings-attachments.cdn.mozilla.net/main-workspace/translations-models/cdea5422-fe0d-4941-a653-ca090423e8f8.spm"
ENJA_SRCVOCAB_SHA256="970c98d174fc01e0339fbabbf45af36a4be3f26f819ec1a5ea1189f71e091889"
ENJA_TRGVOCAB_URL="https://firefox-settings-attachments.cdn.mozilla.net/main-workspace/translations-models/5c1a663f-5736-4100-b681-c6737fe79cb5.spm"
ENJA_TRGVOCAB_SHA256="3b3d9f8f3a034d98d0a476f1794fa79c01e4e98a967ceb6777a66ba2d03ec1e1"
ENJA_LEX_URL="https://firefox-settings-attachments.cdn.mozilla.net/main-workspace/translations-models/87258444-8fac-4d3c-a175-9275e9d10ffa.bin"
ENJA_LEX_SHA256="edfb7eb47b98a2689b804ab3614c31b24f1257aa54b1154da3d85e1ae8152d9f"

download_verified() { # <url> <sha256> <dest>
  local url="$1" sha="$2" dest="$3"
  echo "downloading ${url}"
  curl -fL --retry 3 --retry-delay 2 -o "${dest}" "${url}"
  echo "${sha}  ${dest}" | sha256sum -c -
}

fetch_ende() {
  local dest="${ENDE_DIR}"
  local tarball="${MODELS_DIR}/${ENDE_NAME}.tar.gz"
  if [[ -f "${dest}/model.intgemm.alphas.bin" && -f "${dest}/vocab.deen.spm" && -f "${dest}/lex.s2t.bin" && -f "${dest}/.checked" ]]; then
    echo "ende fixtures present at ${dest}, skipping download"
    return
  fi
  mkdir -p "${dest}" "${MODELS_DIR}"
  download_verified "${ENDE_URL}" "${ENDE_SHA256}" "${tarball}"
  # --strip-components=1 drops the "${ENDE_NAME}/" prefix so overrides of
  # the destination dir extract to the right place too.
  tar -xzf "${tarball}" -C "${dest}" --strip-components=1 \
    "${ENDE_NAME}/model.intgemm.alphas.bin" \
    "${ENDE_NAME}/vocab.deen.spm" \
    "${ENDE_NAME}/lex.s2t.bin"
  rm -f "${tarball}"
  touch "${dest}/.checked"
  echo "ende fixtures ready at ${dest}"
}

fetch_enja() {
  local dest="${ENJA_DIR}"
  if [[ -f "${dest}/model.enja.intgemm.alphas.bin" && -f "${dest}/srcvocab.enja.spm" && -f "${dest}/trgvocab.enja.spm" && -f "${dest}/lex.50.50.enja.s2t.bin" && -f "${dest}/.checked" ]]; then
    echo "enja fixtures present at ${dest}, skipping download"
    return
  fi
  mkdir -p "${dest}"
  download_verified "${ENJA_MODEL_URL}" "${ENJA_MODEL_SHA256}" "${dest}/model.enja.intgemm.alphas.bin"
  download_verified "${ENJA_SRCVOCAB_URL}" "${ENJA_SRCVOCAB_SHA256}" "${dest}/srcvocab.enja.spm"
  download_verified "${ENJA_TRGVOCAB_URL}" "${ENJA_TRGVOCAB_SHA256}" "${dest}/trgvocab.enja.spm"
  download_verified "${ENJA_LEX_URL}" "${ENJA_LEX_SHA256}" "${dest}/lex.50.50.enja.s2t.bin"
  touch "${dest}/.checked"
  echo "enja fixtures ready at ${dest}"
}

fetch_ende
fetch_enja
