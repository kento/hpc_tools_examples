#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=00:15:00
#PJM -L "node=1,freq=2200"
#PJM -j
#PJM -S

. ../env.src

#. ${PREFIX}/${VENV_NAME}/bin/activate

#export HOROVOD_LOG_LEVEL="trace"
#export HOROVOD_LOG_HIDE_TIME=1
#export HOROVOD_CYCLE_TIME=5000
#export HOROVOD_TIMELINE=./timeline_12_async.json
#export HOROVOD_TIMELINE_MARK_CYCLES=1
#export HOROVOD_MPI_THREADS_DISABLE=1
#export OMP_PROC_BIND=false
export OMP_NUM_THREADS=48
#GOMP_CPU_AFFINITY="42-47"

#export HOROVOD_FUSION_THRESHOLD=335544320
#export HOROVOD_FUSION_THRESHOLD=4432
#export HOROVOD_AUTOTUNE=1
#export HOROVOD_AUTOTUNE_LOG=./tunec.csv
#export HOROVOD_CACHE_CAPACITY=1024

ulimit -s 8092

export XOS_MMM_L_HPAGE_TYPE=none
LD_PRELOAD=libtcmalloc.so python3 -u test_train.py --batch 256 --type cpu_mkltensor
