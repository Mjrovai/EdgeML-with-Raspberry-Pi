#!/bin/bash
until grep -qE "SMALL_DONE|Traceback" ~/bench_minicpm_fix_opi.log 2>/dev/null; do sleep 30; done
pkill -x llama-server; sleep 3
python3 ~/bench_memory2.py
