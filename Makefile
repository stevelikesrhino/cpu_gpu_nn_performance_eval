CC = g++

all: nn_eigen

nn_eigen: nn_eigen.cpp
	$(CC) nn_eigen.cpp -o nn_eigen -fopenmp -O3 -march=native -std=c++11 -I/usr/local/include/eigen3

clean:
	rm nn_eigen