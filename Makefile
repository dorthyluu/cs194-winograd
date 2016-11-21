all:
	# g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11
	g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11

clean:
	rm winograd
	rm naive_convolution
