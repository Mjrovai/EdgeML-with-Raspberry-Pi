#!/bin/bash
until grep -q SETUP_DONE ~/setup_bench.log; do sleep 10; done
[ -x ~/llama.cpp-bench/build/bin/llama-server ] || { echo BUILD_FAILED; exit 1; }
ollama ps
python3 ~/bench_small.py rpi
