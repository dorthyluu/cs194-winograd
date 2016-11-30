#!/bin/bash
echo "K C H W" >> bench_filters.txt
for n in {0..9};
do 
    K=$((1 << n))
    echo $((K)) 3 256 256 >> bench_filters.txt
    python3 gen_problem.py $((K)) 3 256 256 > $((K))_3_256_256.in
    ./naive_convolution $((K))_3_256_256.in naive_$((K))_3_256_256.out >> bench_filters.txt
    ./winograd_gpu $((K))_3_256_256.in gpu_$((K))_3_256_256.out >> bench_filters.txt

    echo $((K)) 3 256 256
    ./compare_outputs naive_$((K))_3_256_256.out gpu_$((K))_3_256_256.out

done
