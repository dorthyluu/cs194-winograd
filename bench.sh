#!/bin/bash
# echo "K,CPU,GPU" >> bench.csv
echo "K C H W" >> bench.txt
for n in {3..10};
do 
    K=$((1 << n))
    echo $((K)) 3 258 258 >> bench.txt
    python gen_problem.py $((K)) 3 258 258 > $((K))_3_258_258.in
    ./naive_convolution $((K))_3_258_258.in naive_$((K))_3_258_258.out >> bench.txt
    ./winograd_gpu $((K))_3_258_258.in gpu_$((K))_3_258_258.out >> bench.txt
    # ./winograd_gpu -s 1 -n $n >> bench.csv
done
