all:
	g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11

clean:
	rm winograd
