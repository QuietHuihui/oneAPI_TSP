#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is running OMP TSP.cpp
echo "########## Compiling"
icpx -qopenmp -fopenmp-targets=spir64 ./TSP_SEQ.cpp -o ./TSP_SEQ || exit $?
echo "########## Executing"
./TSP_SEQ
echo "########## Done"
