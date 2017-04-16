CC=g++

ifeq ($(DEBUG),yes)
	CXXFLAGS=-Wall -g -fopenmp
	LDFLAGS=-Wall -g -fopenmp
else
	CXXFLAGS=-Wall -fopenmp
	LDFLAGS=-Wall -fopenmp
endif

default: build

.PHONY: clean cleanall

build: 
	$(CC) assn2.cpp -std=c++11 -fopenmp -larmadillo -llapack -lblas -O3 `pkg-config --cflags --libs opencv` 

run: 
	time ./a.out

clean:
	rm -f *.o

cleanall: clean
	rm -f $(OUT)