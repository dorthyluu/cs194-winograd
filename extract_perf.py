#!/usr/bin/python3
# Parses benchmark output and writes a CSV file.
# Refer to bench_image_size.sh for which programs run

import sys

LINES_PER_BENCH = 3 # num. lines from each program
BENCH_PER_SIZE = 4 # num. benchmarks for each H/W size

BENCH_TEXT_NAME = "benchmark_results/bench_filters.txt"

# skip header line
lines = open(BENCH_TEXT_NAME).read().split("\n")[1:]

out = open("benchmark_results/bench_filters.csv", "w+")
out.write("K,C,H,W,naive,winograd,winograd_openmp,winograd_gpu\n")

index = 0
while index < len(lines) - 1:
  try :
    K, C, H, W = lines[index].split(" ")
  except Exception:
    print("Unexpected format")
    sys.exit(-1)
  index += 1
  mflops = []
  for i in range(BENCH_PER_SIZE * LINES_PER_BENCH):
    line = lines[index + i]
    if "MFlop" in line:
      mflops.append(line.split(" ")[-1])

  if len(mflops) != BENCH_PER_SIZE:
    print("Benchmark output missing values")
    sys.exit(-1)

  outline = "{},{},{},{},".format(K, C, H, W)
  for mflop in mflops:
    outline += mflop + ","

  out.write(outline[:-1] + "\n")
  index += BENCH_PER_SIZE * LINES_PER_BENCH

out.close()
