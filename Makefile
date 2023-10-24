# COMPILER?=nvcc
COMPILER?=/usr/local/cuda-12.2/bin/nvcc -O3 -ccbin /home/lukasm/miniconda3/envs/gcc-test/bin/g++
OPT_FLAGS?=-O3

default: clean compile run

SP_SCAN=sp-scan

compile: $(SP_SCAN)

$(SP_SCAN): utils.cu kernels.cu main.cu
	$(COMPILER) $(OPT_FLAGS) -o $(SP_SCAN).out main.cu

run: $(SP_SCAN)
	./$(SP_SCAN).out $N $B

clean:
	rm -f $(SP_SCAN)
