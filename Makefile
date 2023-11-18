CC_FILES=$(shell find ./ -name "*.cu")
EXE_FILES=$(CC_FILES:.cu=)

all:$(EXE_FILES)

%:%.cu
	nvcc -o $@ $< -O2 -arch=sm_86 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas

clean:
	rm -rf $(EXE_FILES)
