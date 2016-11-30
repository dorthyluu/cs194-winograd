# About:
- We implemented 3x3 convolutions.
- When using the Winograd algorithm for convolutions, we used F(2x2, 3x3), which means that the output for one tile is 2x2.

# OSX setup instructions:
- install latest XCode Command Line Tools
- install [brew](http://brew.sh/)
- install armadillo `brew install armadillo`
- install llvm for openmp `brew install llvm`

# Hive setup instructions:
- Download the source for armadillo
- After unzipping the directory, `cmake .`, then `make`, then `make install DESTDIR=~/lib`

## Generate A Problem
- Compile with `make`
- Create a problem file by running `python3 gen_problem.py > [problem filename]`

## Run Naive Convolution
- Use a file of the generated format (see above) as input for the program `./naive_convolution [input filename] [output filename]`

## Run Winograd Convolution implented serially
- `./winograd [input filename] [output filename]`

## Run Winograd Convolution implemented in OpenMP
- `./winograd_openmp [input filename] [output filename]`

## Run Winograd Convolution implemented in OpenCL
- `./winograd_gpu [input filename] [output filename]`

## Flops Calculation:
- All floating point additions and multiplications are counted as separate operations.
- Manually counted, as Haswell architecture does not support hardware counters

## Benchmarks
- `./bench_image_size.sh`
- `./extract_perf.py BENCHMARK_FILE` will extract MFlop/s and times to `bench_extract_mflops.csv` and `bench_extract_times.csv`.
- NOTE: The benchmark creates many `.in` input files that may take up a lot of disk space. Make sure disk quota does not fill up during benchmarking.
