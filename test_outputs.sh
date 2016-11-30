#!/bin/bash
for n in {2..9};
do 
    N=$((1 << n))
    ./compare_outputs naive_64_3_$((N))_$((N)).out gpu_64_3_$((N))_$((N)).out
done