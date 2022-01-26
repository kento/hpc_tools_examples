#!/bin/bash

./00_tensors.py
./00_tensors_ex.py
./01_datasets.py
./01_datasets_ex.py 
./02_dataloaders.py
./03_build_NN.py    
./04_train_NN.py  
./05_save_NN.py  
./05_save_NN_cleaned.py  
./06_load_NN.py
mpirun -n 4 ./07_dist_train_NN.py
