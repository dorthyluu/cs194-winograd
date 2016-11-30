OBJS = winograd_gpu.o clhelp.o

UNAME_S := $(shell uname -s)

#check if linux
ifeq ($(UNAME_S), Linux)
OCL_INC=-I /usr/local/cuda-6.5/include
OCL_LIB=-L /usr/local/cuda-6.5/lib64 -lOpenCL
# for arma: make install DESTDIR:~/lib
# also requires LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/lib/usr/lib
ARMA_INC= -I ~/lib/usr/include
ARMA_LIB= -L ~/lib/usr/lib -larmadillo 

%.o: %.cpp clhelp.h
	g++ -O2 -c $< $(OCL_INC)

all: $(OBJS)
	g++ $(ARMA_INC) winograd.cpp -o winograd -O2 $(ARMA_LIB) -std=c++11
	g++ $(ARMA_INC) -fopenmp winograd_openmp.cpp -o winograd_openmp -O2 $(ARMA_LIB) -std=c++11
	g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11
	g++ winograd_gpu.o clhelp.o -o winograd_gpu $(OCL_LIB)
endif

#check if os x
ifeq ($(UNAME_S), Darwin)
# openmp on osx requires (brew install llvm)
OPENMP_LIB = -L/usr/local/opt/llvm/lib
OPENMP_INC = -I/usr/local/opt/llvm/include -fopenmp
LLVM_CPP = /usr/local/opt/llvm/bin/clang++

%.o: %.cpp clhelp.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ winograd.cpp -o winograd -O2 -larmadillo -std=c++11
	$(LLVM_CPP) $(OPENMP_INC) winograd_openmp.cpp -o winograd_openmp -O2 $(OPENMP_LIB) -larmadillo -std=c++11
	g++ naive_convolution.cpp -o naive_convolution -O2 -std=c++11
	g++ compare_outputs.cpp -o compare_outputs -O2 -std=c++11
	g++ winograd_gpu.o clhelp.o -o winograd_gpu -framework OpenCL
endif

clean:
	rm -rf $(OBJS) winograd_gpu
	rm winograd
	rm winograd_openmp
	rm naive_convolution
	rm *.in
	rm *.out
