#!/bin/bash
# Reference-code seed distribution (third-party 0.221 code, notebook defaults). 1 thread.
cd "$(dirname "$0")/.." && source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in "$@"; do
  python ct/repro_reference.py --seed $s --threads 1 > logs/ref_seed$s.log 2>&1
done
