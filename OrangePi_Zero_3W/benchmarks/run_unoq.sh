#!/bin/bash
set -x
mkdir -p ~/llama.cpp-bench && cd ~/llama.cpp-bench && git init -q && git remote add origin https://github.com/ggml-org/llama.cpp.git 2>/dev/null
git fetch -q --depth 1 origin 136887b665180c13c6209a4ce0673637b6cd3afd && git checkout -q FETCH_HEAD && git log -1 --format='%h %cd'
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > /dev/null
time cmake --build build -j3 --target llama-bench llama-server
[ -x build/bin/llama-server ] || { echo BUILD_FAILED; exit 1; }
df -h / | tail -1
python3 ~/bench_small.py unoq
