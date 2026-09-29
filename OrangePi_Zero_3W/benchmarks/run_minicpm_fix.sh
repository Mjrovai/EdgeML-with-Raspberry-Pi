#!/bin/bash
# Usage: run_minicpm_fix.sh <board> <file-to-wait-for> <marker>
until grep -qE "$3" "$2" 2>/dev/null; do sleep 30; done
pkill -x llama-server; sleep 3
python3 ~/bench_small.py $1 minicpm server-only
