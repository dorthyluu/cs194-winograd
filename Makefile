all:
	g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11
	g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11

clean:
	rm winograd
	rm naive_convolution
	rm compare_outputs
	rm *.in
	rm *.out

image:
	g++ -o format_image format_image.cpp  -O2 -L/usr/X11R6/lib -lm -lpthread -lX11
	g++ -o recreate_image recreate_image.cpp  -O2 -L/usr/X11R6/lib -lm -lpthread -lX11
