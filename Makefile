OBJS = winograd_gpu.o clhelp.o

UNAME_S := $(shell uname -s)

#check if linux
ifeq ($(UNAME_S), Linux)
OCL_INC=/usr/local/cuda-6.5/include
OCL_LIB=/usr/local/cuda-6.5/lib64
%.o: %.cpp clhelp.h
	g++ -O2 -c $< -I$(OCL_INC)

all: $(OBJS)
	# g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11
	# g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	# g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11
	g++ winograd_gpu.o clhelp.o -o winograd_gpu -L$(OCL_LIB) -lOpenCL
endif

#check if os x
ifeq ($(UNAME_S), Darwin)
%.o: %.cpp clhelp.h
	g++ -O2 -c $<

all: $(OBJS)
	# g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11
	# g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	# g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11
	g++ winograd_gpu.o clhelp.o -o winograd_gpu -framework OpenCL
endif

clean:
	rm winograd
	rm winograd_gpu
	rm naive_convolution
	rm *.in
	rm *.out
