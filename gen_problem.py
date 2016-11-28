import sys
import math
import numpy as np

def gen_problem(K, C, H, W):
	filters = []
	for i in range(K):
		current_filter = []
		filters.append(current_filter)
		for j in range(C):
			current_filter.append(np.random.rand(3, 3))
	return filters, np.random.rand(C, H, W)

if __name__ == "__main__":
	K = 2
	C = 3
	H = 10
	W = 10
	argc = len(sys.argv)
	if (argc != 1 and argc != 5):
		print("".join(["Usage: [python gen_problem.py] to use default values, or ",
			"[python gen_problem.py K C H W] to specify number of filters, number of channels, ",
			"height, and width respectively"]))
		sys.exit()
	if (argc == 5):
		K, C, H, W = tuple([int(el) for el in sys.argv[1:]])
	filters, data = gen_problem(K, C, H, W)
	print(K, C, H, W)
	for _filter in filters:
		for channel in _filter:
			for row in channel:
				print(" ".join([str(el) for el in row]))
			print("")
		print("\n")

	print("\n\n")

	for row in data:
		print(" ".join([str(el) for el in row]))
	print("\n")

