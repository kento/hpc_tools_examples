#export TCSDS_PATH=/opt/FJSVxtclanga/tcsds-1.2.31
#export PYTORCH_PATH=/home/apps/oss/PyTorch-1.7.0
#export LD_LIBRARY_PATH=${TCSDS_PATH}/lib64:${PYTORCH_PATH}/lib:${LD_LIBRARY_PATH}
#export PATH=${TCSDS_PATH}/bin:${PYTORCH_PATH}/bin:${PATH}


export PYTORCH_PATH=/home/apps/oss/PyTorch-1.7.0
export LD_LIBRARY_PATH=${PYTORCH_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${PYTORCH_PATH}/bin:${PATH}
export LD_PRELOAD=/lib64/libfreetype.so.6
