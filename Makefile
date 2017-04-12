CC=g++

ifeq ($(DEBUG),yes)
	CXXFLAGS=-Wall -g -fopenmp
	LDFLAGS=-Wall -g -fopenmp
else
	CXXFLAGS=-Wall -fopenmp
	LDFLAGS=-Wall -fopenmp
endif

default: run

.PHONY: clean cleanall

run: 
	$(CC) assn2.cpp -std=c++11 -larmadillo -llapack -lblas -O3 `pkg-config --cflags --libs opencv` 

clean:
	rm -f *.o

cleanall: clean
	rm -f $(OUT)