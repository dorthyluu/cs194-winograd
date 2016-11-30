# OSX setup instructions:
- install latest XCode Command Line Tools
- install [brew](http://brew.sh/)
- install armadillo `brew install armadillo`
- install llvm for openmp `brew install llvm`

# Hive setup instructions:
- Download the source for armadillo
- After unzipping the directory, `cmake .`, then `make`, then `make install DESTDIR=~/lib`

## Run naive convolution
- Compile with `make`
- Create a problem file by running `python3 gen_problem.py > [problem filename]`
- Use the file as input for the program `./naive_convolution [input filename] [output filename]`

## Flops Calculation:
- All floating point additions and multiplications are counted as separate operations.
- Manually counted, as Haswell architecture does not support hardware counters
