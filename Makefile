all:
	g++ winograd.cpp -o winograd -O2 -larmadillo

clean:
	rm winograd
