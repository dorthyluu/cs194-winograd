#!/bin/bash
echo "K C H W" >> bench_image.txt
for n in {3..9};
do 
    N=$(((1 << n) + 2))
    echo 13 3 $((N)) $((N)) >> bench_image.txt
    python gen_problem.py 13 3 $((N)) $((N)) > 13_3_$((N))_$((N)).in
    ./naive_convolution 13_3_$((N))_$((N)).in naive_13_3_$((N))_$((N)).out >> bench_image.txt
    ./winograd_gpu 13_3_$((N))_$((N)).in gpu_13_3_$((N))_$((N)).out >> bench_image.txt
done
