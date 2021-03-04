CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`


grayscale: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

grayscale-cu: sobel.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm sobel
