#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=00:15:00
#PJM -L "node=1:noncont,freq=2200"
#PJM --mpi "shape=1,proc=4"
#PJM -j
#PJM -S

. ../env.src

export PATH=${TCSDS_PATH}/bin/:${PATH}

#. ${PREFIX}/${VENV_NAME}/bin/activate

#export HOROVOD_LOG_LEVEL="trace"
#export HOROVOD_LOG_HIDE_TIME=1
#export HOROVOD_CYCLE_TIME=5000

#export HOROVOD_TIMELINE=./timeline_12_async.json
#export HOROVOD_TIMELINE_MARK_CYCLES=1
export HOROVOD_MPI_THREADS_DISABLE=1
#export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
#export OMP_NUM_THREADS=48
#GOMP_CPU_AFFINITY="42-47"

#export HOROVOD_FUSION_THRESHOLD=335544320
#export HOROVOD_FUSION_THRESHOLD=4432
#export HOROVOD_AUTOTUNE=1
#export HOROVOD_AUTOTUNE_LOG=./tunec.csv
#export HOROVOD_CACHE_CAPACITY=1024

ulimit -s 8092

################ FJ-MPI
export XOS_MMM_L_HPAGE_TYPE=none
export PLE_MPI_STD_EMPTYFILE="off"
mpirun -np 4 python3 -u pytorch_fapp.py --batch 75 --num-iters 5

