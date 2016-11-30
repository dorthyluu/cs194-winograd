#!/bin/bash
echo "Running..."
echo "K C H W" > bench_image_size.txt
for n in {2..9};
do 
    N=$((1 << n))
    echo 64 3 $((N)) $((N)) >> bench_image_size.txt
    python3 gen_problem.py 64 3 $((N)) $((N)) > 64_3_$((N))_$((N)).in
    ./naive_convolution 64_3_$((N))_$((N)).in naive_64_3_$((N))_$((N)).out >> bench_image_size.txt
    ./winograd 64_3_$((N))_$((N)).in win_64_3_$((N))_$((N)).out >> bench_image_size.txt
    ./winograd_openmp 64_3_$((N))_$((N)).in openmp_64_3_$((N))_$((N)).out >> bench_image_size.txt
    #./winograd_gpu 64_3_$((N))_$((N)).in gpu_64_3_$((N))_$((N)).out >> bench_image_size.txt
done
echo "Comparing outputs..."
./compare_outputs naive_64_3_$((N))_$((N)).out win_64_3_$((N))_$((N)).out
./compare_outputs naive_64_3_$((N))_$((N)).out openmp_64_3_$((N))_$((N)).out
./compare_outputs naive_64_3_$((N))_$((N)).out gpu_64_3_$((N))_$((N)).out
