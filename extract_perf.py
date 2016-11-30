#!/usr/bin/python3
# Parses benchmark output and writes a CSV file.
# Refer to bench_image_*.sh for which programs run

import sys

LINES_PER_BENCH = 3 # num. lines from each program
BENCH_PER_SIZE = 4 # num. benchmarks for each H/W size

argc = len(sys.argv)
if (argc != 2):
  print("Usage: extract_perf.py BENCHMARK_OUTPUT")
  sys.exit(-1)

# skip header line
lines = open(sys.argv[1]).read().split("\n")[1:]

out_flop = open("bench_extract_flops.csv", "w+")
out_time = open("bench_extract_times.csv", "w+")

# write CSV headers
out_flop.write("K,C,H,W,naive,winograd,winograd_openmp,winograd_gpu\n")
out_time.write("K,C,H,W,naive,winograd,winograd_openmp,winograd_gpu\n")

index = 0
while index < len(lines) - 1:
  try :
    K, C, H, W = lines[index].split(" ")
    index += 1
  except Exception:
    print("Unexpected format")
    sys.exit(-1)

  mflops = []
  times = []
  for i in range(BENCH_PER_SIZE * LINES_PER_BENCH):
    line = lines[index + i]
    if "Time" in line:
      times.append(line.split(" ")[-1])
    if "MFlop" in line:
      mflops.append(line.split(" ")[-1])

  if len(mflops) != BENCH_PER_SIZE or len(times) != BENCH_PER_SIZE:
    print("Benchmark output missing values")
    sys.exit(-1)

  outline_flop = "{},{},{},{},".format(K, C, H, W)
  outline_time = "{},{},{},{},".format(K, C, H, W)
  for i in range(len(mflops)):
    outline_flop += mflops[i] + ","
    outline_time += times[i] + ","

  out_flop.write(outline_flop[:-1] + "\n")
  out_time.write(outline_time[:-1] + "\n")
  index += BENCH_PER_SIZE * LINES_PER_BENCH

out_flop.close()
out_time.close()
