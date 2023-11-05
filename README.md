# Single Pass Scan in CUDA
Project for the Programming Massively Parallel Hardware class from University of Copenhagen A.Y. 2023/2024

## Usage 
The program can be ran with Makefile using `make`. To run different experiments modify the Makefile with the desired parameters. The Makefile generate an executable called `sp-scan` that needs 4 arguments to run.

```
./sp-scan <benchmark> <array-length> <block-size> <kernel-version>
```

- `<benchmark>` set benchmark mode, next arguments wil not be considered.
- `<array-length>` size of the input array to test
- `<block-size>` size of one CUDA block
- `<kernel-version>` kernel version from `kernels.cu` file to run:
   - 1 -> Without loopback (spScanKernel)
   - 2 -> Single thread Loopback (spLookbackScanKernel)
   - 3 -> Warp Loopback (spWarpLookbackScanKernel)
   - 4 -> Optimized Warp Loopback (spWarpLookbackScanKernelOpt)


## Authors

- Leonardo Stoppani ([@lilf4p](https://github.com/lilf4p))
- Lukas Mikelionis ([@lukas-mi](https://github.com/lukas-mi))

## References

- Merrill, Duane and Michael Garland. “Single-pass Parallel Prefix Scan with Decoupled Lookback.” (2016).[Nvidia Link](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
