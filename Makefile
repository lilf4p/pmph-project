COMPILER?=nvcc
OPT_FLAGS?=-O3

SP_SCAN=sp-scan

.PHONY: clean all run

default: clean compile run

compile: $(SP_SCAN)

$(SP_SCAN): utils.cu kernels.cu main.cu
	$(COMPILER) $(OPT_FLAGS) -o $(SP_SCAN).out main.cu

run: $(SP_SCAN)
	./$(SP_SCAN).out 0 100003565 512 3
	
clean:
	rm -f $(SP_SCAN)
