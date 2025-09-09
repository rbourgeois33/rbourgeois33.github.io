# Instruction:

```
module load cuda
mkdir build
cd build
cmake -DKokkos_ENABLE_CUDA=ON ..
make -j 20
```

To run an generate a report:
```
ncu --target-processes all --import-source yes --set full -f -o sample-1.ncu-rep ./sample-1
```

You can also build with hip for AMD GPUs !