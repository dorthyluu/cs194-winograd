#!/bin/bash
echo "K C H W" > bench_image.txt
for n in {2..9};
do 
    N=$((1 << n))
    echo 64 3 $((N)) $((N)) >> bench_image.txt
    python3 gen_problem.py 64 3 $((N)) $((N)) > 64_3_$((N))_$((N)).in
    ./naive_convolution 64_3_$((N))_$((N)).in naive_64_3_$((N))_$((N)).out >> bench_image.txt
    ./winograd 64_3_$((N))_$((N)).in winograd_64_3_$((N))_$((N)).out >> bench_image.txt
    ./winograd_openmp 64_3_$((N))_$((N)).in openmp_64_3_$((N))_$((N)).out >> bench_image.txt
    # ./winograd_gpu 64_3_$((N))_$((N)).in gpu_64_3_$((N))_$((N)).out >> bench_image.txt
done
