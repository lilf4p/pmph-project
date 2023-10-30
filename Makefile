COMPILER?=nvcc
OPT_FLAGS?=-O3

SP_SCAN=sp-scan

.PHONY: clean all run

default: clean compile run

compile: $(SP_SCAN)

$(SP_SCAN): utils.cu kernels.cu main.cu wrapper.cu
	$(COMPILER) $(OPT_FLAGS) -o $(SP_SCAN).out main.cu

run: $(SP_SCAN)
	./$(SP_SCAN).out 100003565 256
	# ./$(SP_SCAN).out 16 4 # results in 2 blocks if CHUNK=2
	# ./$(SP_SCAN).out 32 4 # results in 4 blocks  if CHUNK=2

clean:
	rm -f $(SP_SCAN)
