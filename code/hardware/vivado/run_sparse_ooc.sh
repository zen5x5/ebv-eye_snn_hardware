#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 OUTPUT_DIR {sparse_tm|sparse_lm_a} [PART=xczu3eg-sfvc784-2-e] [JOBS=8]" >&2
    exit 2
fi
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_directory="$1"
variant="$2"
target_part="${3:-xczu3eg-sfvc784-2-e}"
jobs="${4:-8}"

if [[ -e "$output_directory" && ! -d "$output_directory" ]]; then
    echo "OUTPUT_DIR exists and is not a directory: $output_directory" >&2
    exit 2
fi
if [[ -d "$output_directory" ]]; then
    shopt -s nullglob dotglob
    output_entries=("$output_directory"/*)
    shopt -u nullglob dotglob
    if (( ${#output_entries[@]} != 0 )); then
        echo "refusing to reuse nonempty OUTPUT_DIR: $output_directory" >&2
        exit 2
    fi
fi
if ! command -v vivado >/dev/null 2>&1; then
    echo "vivado is not available on PATH" >&2
    exit 127
fi

exec vivado -mode batch -nojournal -nolog \
    -source "$script_directory/run_sparse_ooc.tcl" \
    -tclargs "$output_directory" "$variant" "$target_part" "$jobs"
