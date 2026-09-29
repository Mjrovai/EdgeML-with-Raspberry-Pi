#!/bin/bash
until grep -qE "MEMORY_DONE|Traceback" ~/bench_memory.log 2>/dev/null; do sleep 30; done
pkill -x llama-server; sleep 3
python3 ~/bench_context.py
