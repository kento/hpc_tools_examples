#!/bin/bash

make clean_ckpt

command=""
interval=""

mode=$1
case $mode in
    0) ## default run
	command="./pi_with_cr"
	interval="2000000"
	;;
    1) ## pi without cr 
	command="./pi"
	interval=""
	;;
    2) ## pi with cr
	command="./pi_with_cr"
	interval=""
	;;
    *)
	echo "No such test case: $1"
	echo "  0) Default : run.sh (running with ckpt of $interval interval)"
	echo "  1) w/o ckpt: run.sh"
	echo "  2) w/  ckpt: run.sh <interval>"
	exit
	;;
esac

if [ $mode = "2" ]; then
    interval=$2
fi


for i in `seq 10`
do
    id=`ls -t  pi_count.*.ckpt 2> /dev/null | head -n 1 | cut -d "." -f 2`
    if [ -z "$id" ]; then
	# Without ckpt file
	$command 0 $interval & 2> /dev/null
    else
	# with ckpt file
	$command $id $interval & 2> /dev/null
    fi
    proc=$!
    sleep 1
    kill $proc 2> /dev/null
    if [ $? == "1" ]; then
	exit
    fi
    echo "<<<<<<< FAILURE !! >>>>>>>"
done
exit
