#!/bin/bash
# echo "K,CPU,GPU" >> bench.csv
for n in {3..9};
do 
    N=$(((1 << n) + 2))
    ./compare_outputs naive_13_3_$((N))_$((N)).out gpu_13_3_$((N))_$((N)).out
done