#!/bin/bash
echo "K C H W" >> bench_channels.txt
for n in {0..10};
do 
    C=$((1 << n))
    echo 64 $((C)) 256 256 >> bench_channels.txt
    python gen_problem.py 64 $((C)) 256 256 > 64_$((C))_256_256.in
    ./naive_convolution 64_$((C))_256_256.in naive_64_$((C))_256_256.out >> bench_channels.txt
    ./winograd_gpu 64_$((C))_256_256.in gpu_64_$((C))_256_256.out >> bench_channels.txt

    echo 64 $((C)) 256 256
    ./compare_outputs naive_64_$((C))_256_256.out gpu_64_$((C))_256_256.out
done
