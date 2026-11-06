#!/usr/bin/env bash
# Plan action P21: run every model script twice and compare CSV and TXT outputs byte by byte.
# Usage, from scripts/: bash check_reproducibility.sh. Output: resultados/reproducibility.txt.
set -u
PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")"
RESULTS=../resultados
PATTERN='binary_*.csv binary_*.txt f02_vs_n01.csv unseen_mode.csv n01_only*.csv f02_escape.csv label_noise.csv'

run_all() {
  "$PYTHON" train_binary_model.py
  "$PYTHON" train_binary_model.py --no-f02
  "$PYTHON" train_binary_model.py --no-f02 --spectral
  "$PYTHON" analyze_f02_vs_n01.py
  "$PYTHON" evaluate_generalization.py
  "$PYTHON" evaluate_label_noise.py
}

hashes() {
  (cd "$RESULTS" && md5sum $PATTERN)
}

run_all > "$RESULTS/execucao_reproducibility_first.log" 2>&1
hashes > "$RESULTS/.hashes_first.txt"
run_all > "$RESULTS/execucao_reproducibility_second.log" 2>&1
hashes > "$RESULTS/.hashes_second.txt"
{
  echo "files compared: $(wc -l < "$RESULTS/.hashes_first.txt")"
  if diff "$RESULTS/.hashes_first.txt" "$RESULTS/.hashes_second.txt"; then echo "IDENTICAL"; else echo "DIFFERENT"; fi
} > "$RESULTS/reproducibility.txt"
cat "$RESULTS/reproducibility.txt"
