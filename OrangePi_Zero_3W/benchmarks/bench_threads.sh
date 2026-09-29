#!/bin/bash
B=~/llama.cpp/build/bin/llama-bench
M08=~/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf
M2B=~/models/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf
temp() { echo "# temp cpub=$(( $(cat /sys/class/thermal/thermal_zone0/temp)/1000 ))C cpul=$(( $(cat /sys/class/thermal/thermal_zone3/temp)/1000 ))C"; }
for M in $M08 $M2B; do
  echo "######## $(basename $M)"
  temp; echo "== t2 A76 (cpus 6,7)";  taskset -c 6,7   $B -m $M -t 2 -p 512 -n 128 -r 3 -o md 2>/dev/null | grep -E "pp512|tg128"
  temp; echo "== t4 unpinned";        $B -m $M -t 4 -p 512 -n 128 -r 3 -o md 2>/dev/null | grep -E "pp512|tg128"
  temp; echo "== t6 A55 (cpus 0-5)";  taskset -c 0-5   $B -m $M -t 6 -p 512 -n 128 -r 3 -o md 2>/dev/null | grep -E "pp512|tg128"
  temp; echo "== t8 all";             $B -m $M -t 8 -p 512 -n 128 -r 3 -o md 2>/dev/null | grep -E "pp512|tg128"
  temp
done
echo BENCH_DONE
