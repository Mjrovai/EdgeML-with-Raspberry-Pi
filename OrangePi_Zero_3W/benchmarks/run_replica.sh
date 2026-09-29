#!/bin/bash
until grep -q QWEN4B_OK ~/dl2.log; do sleep 10; done
pkill -x llama-server; sleep 3
python3 ~/bench_rpi_replica.py
