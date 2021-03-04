CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`

#ImageProcessor-cpp: ImageProcessor.cpp
#	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

ImageProcessor-cu: ImageProcessor.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm ImageProcessor-* out.jpg
