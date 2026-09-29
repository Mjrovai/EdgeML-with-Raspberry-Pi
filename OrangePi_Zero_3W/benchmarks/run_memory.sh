#!/bin/bash
until grep -qE "CAPTION_DONE|Traceback" ~/bench_caption_opi.log 2>/dev/null; do sleep 30; done
pkill -x llama-server; sleep 3
python3 ~/bench_memory.py
