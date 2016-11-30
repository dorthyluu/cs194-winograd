#!/bin/bash
echo "Running..."
echo "K C H W" > bench_channels.txt
for n in {0..5};
do 
    C=$((1 << n))
    echo 64 $((C)) 512 512 >> bench_channels.txt
    python3 gen_problem.py 64 $((C)) 512 512 > 64_$((C))_512_512.in
    ./naive_convolution 64_$((C))_512_512.in naive_64_$((C))_512_512.out >> bench_channels.txt
    ./winograd 64_$((C))_512_512.in win_64_$((C))_512_512.out >> bench_channels.txt
    ./winograd_openmp 64_$((C))_512_512.in openmp_64_$((C))_512_512.out >> bench_channels.txt
    ./winograd_gpu 64_$((C))_512_512.in gpu_64_$((C))_512_512.out >> bench_channels.txt
done
echo "Comparing output..."
./compare_outputs naive_64_32_512_512.out win_64_32_512_512.out
./compare_outputs naive_64_32_512_512.out openmp_64_32_512_512.out
./compare_outputs naive_64_32_512_512.out gpu_64_32_512_512.out
