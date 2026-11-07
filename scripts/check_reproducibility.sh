#!/usr/bin/env bash
# Run the whole analysis chain (run_all.sh) twice and compare every CSV and TXT output byte by byte.
# Usage, from scripts/: bash check_reproducibility.sh [first|second|compare]. Without an argument, all three phases run.
# Each phase can be run on its own (for instance in separate sessions): `first` and `second` each take a few hours.
# Output: resultados/reproducibility.txt. Set PYTHON if the interpreter is not called python.
set -u
cd "$(dirname "$0")"
RESULTS=../resultados
export PYTHON="${PYTHON:-python}"

hashes() {
  (cd "$RESULTS" && ls *.csv *.txt | grep -v '^reproducibility\.txt$' | xargs md5sum)
}

run_phase() {  # $1 = first | second
  bash ./run_all.sh > "$RESULTS/execucao_reproducibility_$1.log" 2>&1
  hashes > "$RESULTS/.hashes_$1.txt"
}

compare() {
  {
    echo "files compared: $(wc -l < "$RESULTS/.hashes_first.txt")"
    if diff "$RESULTS/.hashes_first.txt" "$RESULTS/.hashes_second.txt"; then echo "IDENTICAL"; else echo "DIFFERENT"; fi
  } > "$RESULTS/reproducibility.txt"
  cat "$RESULTS/reproducibility.txt"
}

case "${1:-all}" in
  first) run_phase first ;;
  second) run_phase second ;;
  compare) compare ;;
  all) run_phase first; run_phase second; compare ;;
  *) echo "Usage: bash check_reproducibility.sh [first|second|compare]"; exit 2 ;;
esac
