#!/bin/bash
echo "Running..."
echo "K C H W" > bench_filters.txt
for n in {0..7};
do 
    K=$((1 << n))
    echo $((K)) 3 512 512 >> bench_filters.txt
    python3 gen_problem.py $((K)) 3 512 512 > $((K))_3_512_512.in
    ./naive_convolution $((K))_3_512_512.in naive_$((K))_3_512_512.out >> bench_filters.txt
    ./winograd $((K))_3_512_512.in win_$((K))_3_512_512.out >> bench_filters.txt
    ./winograd_openmp $((K))_3_512_512.in openmp_$((K))_3_512_512.out >> bench_filters.txt
    ./winograd_gpu $((K))_3_512_512.in gpu_$((K))_3_512_512.out >> bench_filters.txt
done
echo "Comparing outputs..."
./compare_outputs naive_$((K))_3_512_512.out win_$((K))_3_512_512.out
./compare_outputs naive_$((K))_3_512_512.out openmp_$((K))_3_512_512.out
./compare_outputs naive_$((K))_3_512_512.out gpu_$((K))_3_512_512.out
