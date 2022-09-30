#!/bin/bash

for i in $(seq 1 180)
do
  echo "make directory postproc$i"
  mkdir MSDM$i
  touch /workspace/KlausZhangjt.github.io/MSDM$i/"time till now.txt"
  echo microseconds since 1970-01-01 00:00:00 UTC:  >> \
  ./MSDM$i/"time till now.txt"
  cur_ns=`date '+%s%N'`
  echo "scale=0; $cur_ns/1000" | bc >> ./MSDM$i/"time till now.txt"
  # remian integer part
done


#chmod +x ./q1.sh
#./q1.sh   