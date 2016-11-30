#!/bin/bash
echo "K C H W" >> bench_channels.txt
for n in {0..5};
do 
    C=$((1 << n))
    echo 64 $((C)) 512 512 >> bench_channels.txt
    python3 gen_problem.py 64 $((C)) 512 512 > 64_$((C))_512_512.in
    ./naive_convolution 64_$((C))_512_512.in naive_64_$((C))_512_512.out >> bench_channels.txt
    ./winograd_gpu 64_$((C))_512_512.in gpu_64_$((C))_512_512.out >> bench_channels.txt

    echo 64 $((C)) 512 512
    ./compare_outputs naive_64_$((C))_512_512.out gpu_64_$((C))_512_512.out
done
