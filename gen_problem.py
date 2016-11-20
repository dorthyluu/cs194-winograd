import sys
import math
import numpy as np

def gen_problem(K, C, H, W, N):
	filters = []
	for i in range(K):
		current_filter = []
		filters.append(current_filter)
		for j in range(C):
			current_filter.append(np.random.rand(3, 3))
	data = []
	for i in range(N):
		current_data = []
		data.append(current_data)
		for j in range(C):
			current_data.append(np.random.rand(H, W))
	return filters, data

if __name__ == "__main__":
	K = 2
	C = 3
	H = 10
	W = 10
	N = 2
	argc = len(sys.argv)
	if (argc != 1 and argc != 6):
		print("".join(["Usage: [python gen_problem.py] to use default values, or ",
			"[python gen_problem.py K C H W N] to specify number of filters, number of channels, ",
			"height, width, and number of images respectively"]))
		sys.exit()
	if (argc == 6):
		K, C, H, W, N = tuple([int(el) for el in sys.argv[1:]])
	filters, data = gen_problem(K, C, H, W, N)
	print(K, C, H, W, N)
	for _filter in filters:
		for channel in _filter:
			for row in channel:
				print(" ".join([str(el) for el in row]))
			print("")
		print("\n")

	print("\n\n")

	for image in data:
		for channel in image:
			for row in channel:
				print(" ".join([str(el) for el in row]))
			print("")
		print("\n")

