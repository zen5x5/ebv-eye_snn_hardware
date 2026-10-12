#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "usage: $0 OUTPUT_DIR [FULL_STATE=0] [PART=xczu3eg-sfvc784-2-e] [JOBS=8]" >&2
    exit 2
fi
if ! command -v vivado >/dev/null 2>&1; then
    echo "vivado is not available on PATH" >&2
    exit 127
fi

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_directory="$1"
full_state="${2:-0}"
target_part="${3:-xczu3eg-sfvc784-2-e}"
jobs="${4:-8}"

exec vivado -mode batch -nojournal -nolog \
    -source "$script_directory/run_ooc.tcl" \
    -tclargs "$output_directory" "$full_state" "$target_part" "$jobs"
