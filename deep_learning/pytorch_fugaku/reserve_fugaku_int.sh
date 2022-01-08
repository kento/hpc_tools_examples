#!/bin/bash
pjsub --interact -L "node=1" -L "rscgrp=int" -L "elapse=3:00:00" --sparam "wait-time=600" --mpi "proc=8"
