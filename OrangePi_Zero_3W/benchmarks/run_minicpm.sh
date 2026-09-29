#!/bin/bash
# Usage: run_minicpm.sh <board> [file-to-wait-for] [marker]
BOARD=$1
[ -n "$2" ] && until grep -qE "$3" "$2" 2>/dev/null; do sleep 30; done
for s in 1B 2B; do
  mkdir -p ~/models/MiniCPM5-$s-Q4_K_M
  (cd ~/models/MiniCPM5-$s-Q4_K_M && wget -q -c https://huggingface.co/openbmb/MiniCPM5-$s-GGUF/resolve/main/MiniCPM5-$s-Q4_K_M.gguf)
done
ls -l ~/models/MiniCPM5-*/
pkill -x llama-server; sleep 3
python3 ~/bench_small.py $BOARD minicpm
