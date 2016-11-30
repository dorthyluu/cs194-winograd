all:
	g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11
	g++ fft_convolution.cpp -o fft_convolution -O2 -larmadillo -std=c++11
	g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11

clean:
	rm winograd
	rm fft_convolution
	rm naive_convolution
	rm *.in
	rm *.out
