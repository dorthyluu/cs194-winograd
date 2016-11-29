#!/bin/bash
# echo "K,CPU,GPU" >> bench.csv
for n in {3..9};
do 
    K=$((1 << n))
    ./compare_outputs naive_$((K))_3_258_258.out gpu_$((K))_3_258_258.out
done
