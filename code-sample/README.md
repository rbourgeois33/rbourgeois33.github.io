# Instruction:
Load `cuda` or `rocm`. Then in build e.g. `cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/cuda.ada.cmake`, you can easily make a custom toolchain file

To run an generate a `ncu` report:
```
ncu --target-processes all --import-source yes --set full -f -o sample-1.ncu-rep ./sample-1
```

