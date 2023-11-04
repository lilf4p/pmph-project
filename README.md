# Single Pass Scan in CUDA
Project for the PMPH class from University of Copenhagen A.Y. 2023/2024

## Plan 
Each thread process several element Q. Block size B. One block process B*Q elements 
1. Collective read/write of B*Q elements from glb2sh2reg/reg2sh2glob (coalesced) -> compare performance with memcpy
2. Insert dynamic BlockID
3. Block-level scan -> check performance
   1. each thread scans its Q elems in regs
   2. writes to shared the last elems 
   3. block level scan
   4. each th select the corresp prefix in shared mem and adds it to each of its saved elems 


Graphs: 
   - BLOCK SIZES fixed input sizes (221184, 1000000, 100003565) and fixed chunk (15)
   - CHUNK SIZES fixed input sizes (221184, 1000000, 100003565) and fixed block (512)


## Authors

- Leonardo Stoppani ([@lilf4p](https://github.com/lilf4p))
- Lukas Mikelionis ([@lukas-mi](https://github.com/lukas-mi))

## References

- Merrill, Duane and Michael Garland. “Single-pass Parallel Prefix Scan with Decoupled Lookback.” (2016).[Nvidia Link](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
