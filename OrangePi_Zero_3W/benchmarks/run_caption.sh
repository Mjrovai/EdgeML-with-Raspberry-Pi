#!/bin/bash
# Usage: run_caption.sh <board>   (waits for the MiniCPM5 run on this board to finish)
BOARD=$1
until grep -qE "SMALL_DONE|Traceback" ~/bench_minicpm_$BOARD.log 2>/dev/null; do sleep 30; done
U=https://huggingface.co/unsloth
case $BOARD in
  rpi)  cd ~/models/bench-mtp && wget -q -c -O mmproj-0.8B-F16.gguf $U/Qwen3.5-0.8B-MTP-GGUF/resolve/main/mmproj-F16.gguf \
                              && wget -q -c -O mmproj-2B-F16.gguf  $U/Qwen3.5-2B-MTP-GGUF/resolve/main/mmproj-F16.gguf ;;
  unoq) cd ~/models/Qwen3.5-0.8B-MTP-GGUF && wget -q -c $U/Qwen3.5-0.8B-MTP-GGUF/resolve/main/mmproj-F16.gguf ;;
esac
until [ "$BOARD" != opi ] || grep -q MMGEMMA_OK ~/dl_mmproj.log; do sleep 30; done
pkill -x llama-server; sleep 3
python3 ~/bench_caption.py $BOARD
