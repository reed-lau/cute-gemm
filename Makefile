CC_FILES=$(shell find ./ -name "*.cu")
EXE_FILES=$(CC_FILES:.cu=)

.PHONY: all clean gemm

all:$(EXE_FILES)

gemm: gemm-multi-stage gemm-simple

%:%.cu
	nvcc -o $@ $< -O2 -arch=sm_86 -std=c++17 -I3rd/cutlass/include --expt-relaxed-constexpr -cudart shared --cudadevrt none -lcublasLt -lcublas

clean:
	rm -rf $(EXE_FILES)
