#!/bin/bash
until grep -qE "REPLICA_DONE|Traceback" ~/bench_rpi_replica.log; do sleep 30; done
pkill -x llama-server; sleep 3
python3 ~/bench_small.py opi
