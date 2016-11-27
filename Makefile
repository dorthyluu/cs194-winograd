# OpenCL on OS X
all:
	g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11
	g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11
	g++ -O2 -c winograd_gpu.cpp
	g++ -O2 -c clhelp.cpp
	g++ winograd_gpu.o clhelp.o -o winograd_gpu -framework OpenCL

clean:
	rm winograd
	rm naive_convolution
	rm *.in
	rm *.out
