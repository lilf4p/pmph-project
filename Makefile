COMPILER?=nvcc
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
