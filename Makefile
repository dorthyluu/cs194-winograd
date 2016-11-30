OBJS = winograd_gpu.o clhelp.o

UNAME_S := $(shell uname -s)

#check if linux
ifeq ($(UNAME_S), Linux)
OCL_INC=-I /usr/local/cuda-6.5/include
OCL_LIB=-L /usr/local/cuda-6.5/lib64 -lOpenCL
# for arma: make install DESTDIR:~/lib
ARMA_INC= -I ~/lib/usr/include
ARMA_LIB= -L ~/lib/usr/lib -larmadillo 

%.o: %.cpp clhelp.h
	g++ -O2 -c $< $(OCL_INC)

all: $(OBJS)
	g++ $(ARMA_INC) winograd.cpp -o winograd -O2 $(ARMA_LIB) -std=c++11
	g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11
	g++ winograd_gpu.o clhelp.o -o winograd_gpu $(OCL_LIB)
endif

#check if os x
ifeq ($(UNAME_S), Darwin)
%.o: %.cpp clhelp.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11
	g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11
	g++ winograd_gpu.o clhelp.o -o winograd_gpu -framework OpenCL
endif

clean:
	rm -rf $(OBJS) winograd_gpu
	rm winograd
	rm naive_convolution
	rm compare_outputs
	rm *.in
	rm *.out

image:
	g++ -o format_image format_image.cpp  -O2 -L/usr/X11R6/lib -lm -lpthread -lX11
	g++ -o recreate_image recreate_image.cpp  -O2 -L/usr/X11R6/lib -lm -lpthread -lX11
